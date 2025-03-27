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
        self.has_ongoing_session = False

        # TODO: Pass camera ID then open that
        self.open_camera()

    def update_session_status(self):
        """Updates the session status"""
        self.has_ongoing_session = self.database_queries.get_settings()[
            "has_ongoing_session"]

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
        while True:
            ret, frame = self.cap.read()
            if not ret:
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

    def open_camera(self):
        """Opens camera"""
        self.cap = cv2.VideoCapture(0)
        # If cannot open camera, exit
        if not self.cap.isOpened():
            print("❌ Unable to open camera!")
            exit()
