import cv2
import torch
import asyncio
import websockets
import base64
import json
import subprocess
from collections import deque
from frame_sequences.hybrid_model import HybridModel
from PIL import Image

# ───── CONFIGURATION ───────────────────────────────────────────────
_TRAINED_MODEL_SAVE_PATH = "./saved_models/disdrive_model.pth"
_WEBSERVER_PATH = "./web_server"
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

# ───── MODEL INITIALIZATION ─────────────────────────────────────────


def load_model():
    model = HybridModel()
    model.load_state_dict(torch.load(
        _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
    model.to(_DEVICE)
    model.eval()
    return model


model = load_model()
frame_buffer = deque(maxlen=20)

# ───── SHARED STATE ────────────────────────────────────────────────
latest_data = {"frame": None, "behavior": "Detecting..."}
clients = set()

# ───── CAMERA SETUP ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open camera!")
    exit()

# ───── FEATURE EXTRACTION ──────────────────────────────────────────


def extract_features(frame):
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = Image.fromarray(processed_frame)
    processed_frame = model.preprocessor(
        processed_frame).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        features = model.clip_model.encode_image(
            processed_frame).squeeze(0).to(torch.float32)

    return features

# ───── DETECTION LOOP (RUNS IN BACKGROUND) ─────────────────────────


async def detection_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        feature = extract_features(frame)
        frame_buffer.append(feature)

        behavior = "Detecting..."
        if len(frame_buffer) == 20:
            with torch.no_grad():
                sequence_tensor = torch.stack(list(frame_buffer)).unsqueeze(0)
                output = model(sequence_tensor)
                output = torch.argmax(output, dim=1).item()
                behavior = _BEHAVIOR_LABEL[output]

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        # Update shared state
        latest_data["frame"] = frame_bytes
        latest_data["behavior"] = behavior

        await asyncio.sleep(0.01)

# ───── WEBSOCKET CLIENT HANDLER ────────────────────────────────────


async def video_stream(websocket):
    clients.add(websocket)
    print("📡 Client connected")
    try:
        while True:
            if latest_data["frame"] is not None:
                await websocket.send(json.dumps(latest_data))
            await asyncio.sleep(0.01)
    except websockets.ConnectionClosed:
        print("🔌 Client disconnected")
    finally:
        clients.remove(websocket)

# ───── MAIN ENTRY POINT ────────────────────────────────────────────


async def main():
    # Start detection loop
    asyncio.create_task(detection_loop())

    # Start WebSocket server
    async with websockets.serve(video_stream, "0.0.0.0", 8765):
        print("✅ WebSocket server started on ws://0.0.0.0:8765")

        # Start React frontend
        try:
            print("🚀 Starting React frontend...")
            subprocess.Popen(["npm", "run", "dev"],
                             cwd=_WEBSERVER_PATH, shell=True)
        except Exception as e:
            print(f"⚠️ Failed to start React frontend: {e}")

        await asyncio.Future()  # Keep running forever

# ───── RUN APP ─────────────────────────────────────────────────────
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.run_forever()
