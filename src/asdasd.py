import cv2
import torch
import asyncio
import websockets
import base64
import json
import subprocess
from collections import deque
from frame_sequences.hybrid_model_original import HybridModel
from PIL import Image

# Model settings
_TRAINED_MODEL_SAVE_PATH = "./saved_models/disdrive_hybrid_weights.pth"
_WEBSERVER_PATH = "./web_server"  # Path to React frontend
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BEHAVIOR_LABEL = {
    0: "Safe Driving",
    1: "Texting Right",
    2: "Texting Left",
    3: "Talking using Phone Right",
    4: "Talking using Phone Left",
    5: "Drinking",
    6: "Head Down",
    7: "Look Behind",
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to open camera!")
    exit()

# Load and initialize the model


def load_model():
    model = HybridModel()
    model.load_state_dict(torch.load(
        _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
    model.to(_DEVICE)
    model.eval()  # Set to evaluation mode
    return model


model = load_model()
frame_buffer = deque(maxlen=20)  # Buffer to store 20 frames


def extract_features(frame):
    """Extracts features from a frame using CLIP"""
    processed_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    processed_frame = Image.fromarray(processed_frame)  # Convert to PIL Image
    processed_frame = model.preprocessor(
        processed_frame).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():  # Extract features using CLIP
        features = model.clip_model.encode_image(
            processed_frame).squeeze(0).to(torch.float32)

    return features


async def video_stream(websocket):
    """WebSocket server handling video streaming"""
    print(f"✅ Client connected: {websocket.remote_address}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame!")
            break

        # Extract features
        feature = extract_features(frame)
        frame_buffer.append(feature)

        # Make prediction if 20 frames are collected
        predicted_behavior = "Detecting..."
        if len(frame_buffer) == 20:
            with torch.no_grad():
                sequence_tensor = torch.stack(list(frame_buffer)).unsqueeze(
                    0)  # Convert buffer to tensor
                output = model(sequence_tensor)
                # Get predicted class
                output = torch.argmax(output, dim=1).item()
                predicted_behavior = _BEHAVIOR_LABEL[output]

        # Encode frame to base64 for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        # Send JSON data to frontend
        message = json.dumps(
            {"frame": frame_bytes, "behavior": predicted_behavior})
        await websocket.send(message)

        await asyncio.sleep(0.05)  # 50ms delay for smooth streaming


async def start_websocket():
    """Start the WebSocket server"""
    async with websockets.serve(video_stream, "0.0.0.0", 8765):
        print("✅ WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # Keep running indefinitely
        print("1")

    print("2")


def start_frontend():
    """Start the React frontend (Optional)"""
    try:
        print("🚀 Starting React frontend...")
        subprocess.Popen(["npm", "run", "dev"],
                         cwd=_WEBSERVER_PATH, shell=True)
    except Exception as e:
        print(f"⚠️ Failed to start React frontend: {e}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start backend first
    loop.run_until_complete(start_websocket())

    # Uncomment the line below if you want to auto-start React as well
    # start_frontend()

    loop.run_forever()  # Keep backend running
