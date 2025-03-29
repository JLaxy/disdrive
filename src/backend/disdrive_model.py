import cv2
import torch
import asyncio
import base64
import time
from collections import deque
from backend.log_manager import LogManager
from frame_sequences.hybrid_model import HybridModel
from PIL import Image
from backend.database_queries import DatabaseQueries
import threading

# Fix GPU usage - IMPORTANT CHANGE
_TRAINED_MODEL_SAVE_PATH = "./saved_models/disdrive_model.pth"
_DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
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

# Configuration for performance tuning
_FRAME_SKIP = 2  # Process every Nth frame
_FRAME_WIDTH = 224  # Resize width before processing
_FRAME_HEIGHT = 224  # Resize height before processing


class DisdriveModel:
    """Handles all functionalities related to the Machine Learning Model"""

    def __init__(self, database_queries: DatabaseQueries):
        print(f"Using device: {_DEVICE}")
        self.model = HybridModel()
        self.model.load_state_dict(torch.load(
            _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
        self.model.to(_DEVICE)
        self.model.eval()

        # Preload and cache preprocessor transforms
        if hasattr(self.model, 'clip_model') and hasattr(self.model, 'preprocessor'):
            print("Warming up CLIP model...")
            # Warmup the model with a dummy inference
            dummy_input = torch.zeros(1, 3, 224, 224).to(_DEVICE)
            with torch.no_grad():
                self.model.clip_model.encode_image(dummy_input)

        self.frame_buffer = deque(maxlen=20)
        self.latest_detection_data = {
            "frame": None, "behavior": "Detecting..."}
        self.database_queries = database_queries
        self.log_manager = LogManager(self.database_queries)
        self.update_session_status()

        # Camera-related attributes
        self.available_cameras = self.detect_cameras()
        print(f"Available cameras: {self.available_cameras}")
        self.current_camera_index = 0
        self.cap = None

        # FPS tracking attributes
        self.frame_count = 0
        self.fps_start_time = None
        self.current_fps = 0

        # Frame processing attributes
        self.processed_frames = 0
        self.frame_skip_counter = 0

        # Threading for feature extraction
        self.feature_queue = asyncio.Queue(maxsize=5)
        self.frame_queue = asyncio.Queue(maxsize=5)

        saved_camera = self.get_selected_camera_saved()
        self._detection_loop_task = None
        self._feature_extraction_task = None

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

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution capture
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30fps if supported

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
        # Start timing the process
        start_time = time.time()

        # Resize frame to improve performance
        resized_frame = cv2.resize(frame, (_FRAME_WIDTH, _FRAME_HEIGHT))

        # Convert to RGB for PIL
        processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        processed_frame = Image.fromarray(processed_frame)

        # Use the CLIP preprocessor
        processed_frame = self.model.preprocessor(
            processed_frame).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            features = self.model.clip_model.encode_image(
                processed_frame).squeeze(0).to(torch.float32)

        # Print time taken for feature extraction
        extraction_time = time.time() - start_time
        print(f"Feature extraction took: {extraction_time:.4f} seconds")

        return features

    async def feature_extraction_worker(self):
        """Worker to extract features asynchronously"""
        try:
            while True:
                if asyncio.current_task().cancelled():
                    print("Feature extraction worker cancelled")
                    break

                # Get frame from queue
                frame = await self.frame_queue.get()

                # Use ThreadPoolExecutor for CPU-bound operations
                loop = asyncio.get_event_loop()
                feature = await loop.run_in_executor(None, self.extract_features, frame)

                # Put feature in queue for main loop
                await self.feature_queue.put(feature)

                # Mark task as done
                self.frame_queue.task_done()

        except asyncio.CancelledError:
            print("Feature extraction worker was cancelled")
        except Exception as e:
            print(f"Error in feature extraction worker: {e}")

    async def detection_loop(self):
        """Responsible for detecting behavior of driver"""
        print("Starting Detection...")
        self.fps_start_time = asyncio.get_event_loop().time()
        self.frame_count = 0
        self.frame_skip_counter = 0

        try:
            # Start feature extraction worker
            self._feature_extraction_task = asyncio.create_task(
                self.feature_extraction_worker())

            while True:
                # Check if task will be cancelled
                if asyncio.current_task().cancelled():
                    print("Detection loop cancelled")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    await asyncio.sleep(0.1)
                    continue

                # Count each successfully captured frame
                self.frame_count += 1

                # Calculate and print FPS every 3 seconds
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - self.fps_start_time

                if elapsed_time >= 3.0:
                    self.current_fps = self.frame_count / elapsed_time
                    print(
                        f"Current FPS: {self.current_fps:.2f} | Processed frames: {self.processed_frames}")
                    self.frame_count = 0
                    self.processed_frames = 0
                    self.fps_start_time = current_time

                # Always encode and update the visual feed for the UI
                # Use a smaller resolution for the encoded frame
                display_frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', display_frame, [
                                         cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                self.latest_detection_data["frame"] = frame_bytes
                self.latest_detection_data["fps"] = f"{self.current_fps:.1f}"

                # If to not detect, skip detection
                if not self.has_ongoing_session:
                    self.latest_detection_data["behavior"] = "Detection Paused"

                    if self.log_manager.has_session:
                        self.log_manager.end_session()

                    await asyncio.sleep(0.01)
                    continue

                # If still not started, then start
                if not self.log_manager.has_session:
                    self.log_manager.start_session()

                # Process only every Nth frame for performance
                if self.frame_skip_counter % _FRAME_SKIP == 0:
                    # Queue the frame for feature extraction
                    if not self.frame_queue.full():
                        await self.frame_queue.put(frame.copy())
                        self.processed_frames += 1

                    # Try to get a feature from the queue
                    try:
                        feature = self.feature_queue.get_nowait()
                        self.frame_buffer.append(feature)
                        self.feature_queue.task_done()
                    except asyncio.QueueEmpty:
                        # No features available yet, continue
                        pass

                self.frame_skip_counter += 1

                # Process the frame buffer for behavior prediction
                behavior = "Detecting..."
                if len(self.frame_buffer) == 20:
                    with torch.no_grad():
                        # Start timing inference
                        inference_start = time.time()

                        # Create tensor from buffer
                        sequence_tensor = torch.stack(
                            list(self.frame_buffer)).unsqueeze(0).to(_DEVICE)

                        # Run model inference
                        output = self.model(sequence_tensor)
                        output = torch.argmax(output, dim=1).item()
                        behavior = _BEHAVIOR_LABEL[output]

                        # Print inference time
                        inference_time = time.time() - inference_start
                        print(
                            f"Behavior inference took: {inference_time:.4f} seconds")

                # If behavior changed from previous behavior
                if self.latest_detection_data["behavior"] != behavior:
                    self.log_manager.end_behavior()
                    # store behavior start
                    self.log_manager.new_behavior_started(behavior)

                # Update shared state for all clients to access
                self.latest_detection_data["behavior"] = behavior

                # Sleep to prevent busy-waiting but keep it minimal
                await asyncio.sleep(0.005)

        except asyncio.CancelledError:
            print("Detection loop was cancelled")
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            # Cancel feature extraction worker
            if self._feature_extraction_task:
                self._feature_extraction_task.cancel()
                try:
                    await self._feature_extraction_task
                except asyncio.CancelledError:
                    pass

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

            # Cancel existing detection loop and feature extraction if running
            if hasattr(self, '_detection_loop_task') and self._detection_loop_task and not self._detection_loop_task.done():
                print("Cancelling existing detection loop...")
                self._detection_loop_task.cancel()
                try:
                    await self._detection_loop_task
                except asyncio.CancelledError:
                    print("Previous detection loop cancelled successfully")

            if hasattr(self, '_feature_extraction_task') and self._feature_extraction_task and not self._feature_extraction_task.done():
                self._feature_extraction_task.cancel()
                try:
                    await self._feature_extraction_task
                except asyncio.CancelledError:
                    print("Feature extraction task cancelled successfully")

            # Clear queues
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            while not self.feature_queue.empty():
                try:
                    self.feature_queue.get_nowait()
                    self.feature_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # Release existing camera if open
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(camera_id)

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Validate new camera capture
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            self.frame_buffer.clear()

            # Reset FPS tracking
            self.frame_count = 0
            self.fps_start_time = None
            self.current_fps = 0
            self.processed_frames = 0
            self.frame_skip_counter = 0

            # Update current camera index
            self.current_camera_index = camera_id

            # Restart loop
            self._detection_loop_task = asyncio.create_task(
                self.detection_loop())

            print(f"Successfully changed to camera {camera_id}")

        except Exception as e:
            print(f"Error changing camera: {e}")
