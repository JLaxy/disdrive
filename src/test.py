"""Testing Model with camera"""

import cv2
import torch
import numpy as np
from frame_sequences.hybrid_model import HybridModel
from collections import deque
from PIL import Image

_TRAINED_MODEL_SAVE_PATH = "./saved_models/BEST_3zero.pth"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def __load_model() -> HybridModel:
    """Loads and initializes model"""
    CLIP_LSTM: HybridModel = HybridModel()
    # Load saved model form disk
    CLIP_LSTM.load_state_dict(torch.load(_TRAINED_MODEL_SAVE_PATH))
    CLIP_LSTM.to(_DEVICE)  # Move Hybrid Model to device

    return CLIP_LSTM


def __open_camera() -> cv2.VideoCapture:
    """Opens webcam"""
    cam = cv2.VideoCapture(0)  # Open camera
    if not cam.isOpened():
        print("Unable to open camera!")
        exit()

    return cam


def __extract_features(frame):
    """Preprocesses frame then extracts feature from frame"""
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    processed_frame = Image.fromarray(processed_frame)  # Convert to PIL Image
    # Preprocess frame then move to device
    processed_frame = model.preprocessor(
        processed_frame).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():  # Extract features using CLIP
        features = model.clip_model.encode_image(
            processed_frame).to(torch.float32)

    return features  # Return feature


if __name__ == "__main__":
    model = __load_model()  # Load model
    cam = __open_camera()  # Open camera

    frame_buffer = deque(maxlen=20)  # Where frames will be held

    while True:
        ret, frame = cam.read()  # Get current frame
        if not ret:  # If no frame is captured
            raise RuntimeError("Failed to capture a frame!")

        # Preprocess frame then extract features
        feature = __extract_features(frame)

        frame_buffer.append(feature)  # Add to frame buffer

        if len(frame_buffer) == 20:  # If collected 20 frames
            # Convert to list then to Tensor
            sequence_tensor = torch.stack(list(frame_buffer))
            print(f"Test: {sequence_tensor.shape}")

            output = model(sequence_tensor)  # Make prediction
            print(output)

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break
