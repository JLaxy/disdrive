import cv2
import torch
import asyncio
import base64
from collections import deque
from frame_sequences.hybrid_model import HybridModel
from PIL import Image
from backend.database_queries import DatabaseQueries

_TRAINED_MODEL_SAVE_PATH = "./saved_models/disdrive_model.pth"
_WEBSERVER_PATH = "./web_server"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BEHAVIOR_LABEL = {
    0: "Safe Driving",
    1: "Texting",
    2: "Texting",
    3: "Talking using Phone",
    4: "Talking using Phone",
    5: "Drinking",
    6: "Head Down",
    7: "Look Behind",
}


class DisdriveModel:
    """Handles all functionalities related to the Machine Learning Model"""

    def __init__(self, database_queries: DatabaseQueries):
        self.model = HybridModel()
        self.model.load_state_dict(torch.load(
            _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
        self.model.to(_DEVICE)
        self.model.eval()

        self.frame_buffer = deque(maxlen=20)
        self.latest_detection_data = {
            "frame": None, "behavior": "Detecting..."}
        self.database_queries = database_queries
        self.update_session_status()

        # Camera-related attributes
        self.available_cameras = self.detect_cameras()

        print(f"Available cameras: {self.available_cameras}")
        self.current_camera_index = 0
        self.cap = None

        saved_camera = self.get_selected_camera_saved()

        self._detection_loop_task = None

        # If no saved camera or saved camera is available
        if saved_camera == None or (saved_camera not in self.available_cameras):
            print(
                f"DISDRIVE_MODEL: Saved camera with index {saved_camera} not available! opening nearest available camera...")
            # Open first available camera
            self.open_camera(self.available_cameras[0])
        else:
            self.open_camera(saved_camera)

    def detect_cameras(self):
        """
        Detect available cameras on the system.

        Returns:
        list: A list of dictionaries containing camera information
        """
        available_cameras = []
        max_cameras_to_check = 5  # Limit the number of cameras to check

        for index in range(max_cameras_to_check):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()

        return available_cameras

    def get_selected_camera_saved(self):
        return self.database_queries.get_settings()["camera_id"]

    def open_camera(self, camera_index):
        """
        Open a specific camera by its index.

        Args:
        camera_index (int): Index of the camera to open
        """
        # Release existing camera if open
        if self.cap is not None:
            self.cap.release()

        print(f"DISDRIVE_MODEL: opening camera with index {camera_index}")

        # Open new camera
        self.cap = cv2.VideoCapture(camera_index)
        self.current_camera_index = camera_index

        # If cannot open camera, print error
        if not self.cap.isOpened():
            print(f"❌ Unable to open camera with index {camera_index}!")
            self.current_camera_index = None
        else:
            # Update opened camera
            self.database_queries.update_setting("camera_id", camera_index)

    def update_session_status(self):
        """Updates the session status"""
        self.has_ongoing_session = bool(self.database_queries.get_settings()[
            "has_ongoing_session"])

    def extract_features(self, frame):
        """Extracts features of retrieved frame from camera"""
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = Image.fromarray(processed_frame)
        processed_frame = self.model.preprocessor(
            processed_frame).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            features = self.model.clip_model.encode_image(
                processed_frame).squeeze(0).to(torch.float32)

        return features

    async def detection_loop(self):
        """Responsible for detecting behavior of driver"""
        print("Starting Detection...")

        try:
            while True:

                # Check if task will be cancelled
                if asyncio.current_task().cancelled():
                    print("Detection loop cancelled")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    # Optionally, add a short delay or break
                    await asyncio.sleep(0.1)
                    continue

                # Encode frame to base64 for transmission
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                self.latest_detection_data["frame"] = frame_bytes

                # If to not detect, skip detection
                if not self.has_ongoing_session:
                    self.latest_detection_data["behavior"] = "Detection Paused"
                    await asyncio.sleep(0.01)  # Prevent busy-waiting
                    continue

                # Extract features using CLIP encoder
                feature = self.extract_features(frame)
                self.frame_buffer.append(feature)

                behavior = "Detecting..."
                if len(self.frame_buffer) == 20:
                    with torch.no_grad():
                        sequence_tensor = torch.stack(
                            list(self.frame_buffer)).unsqueeze(0)
                        output = self.model(sequence_tensor)
                        output = torch.argmax(output, dim=1).item()
                        behavior = _BEHAVIOR_LABEL[output]

                # Update shared state for all clients to access
                self.latest_detection_data["behavior"] = behavior
                await asyncio.sleep(0.01)  # Prevent busy-waiting

        except asyncio.CancelledError:
            print("Detection loop was cancelled")
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            # Ensure camera is released
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    async def change_camera(self, camera_id):
        """Changes camera used by the model safely"""
        try:
            if camera_id not in self.available_cameras:
                raise ValueError(
                    f"Camera {camera_id} not in available cameras")

            # Cancel existing detection loop if running
            if hasattr(self, '_detection_loop_task') and self._detection_loop_task and not self._detection_loop_task.done():
                print("Cancelling existing detection loop...")
                self._detection_loop_task.cancel()
                try:
                    await self._detection_loop_task
                except asyncio.CancelledError:
                    print("Previous detection loop cancelled successfully")

            # Release existing camera if open
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(camera_id)

            # Validate new camera capture
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            self.frame_buffer.clear()

            # Update current camera index
            self.current_camera_index = camera_id

            # Restart loop
            self._detection_loop_task = asyncio.create_task(
                self.detection_loop())

            print(f"Successfully changed to camera {camera_id}")

        except Exception as e:
            print(f"Error changing camera: {e}")
