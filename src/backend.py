import cv2
import torch
import asyncio
import websockets
import base64
import json
import subprocess
import sys
from collections import deque
from frame_sequences.hybrid_model import HybridModel
from PIL import Image
from urllib.parse import parse_qs

# ───── CONFIGURATION ───────────────────────────────────────────────

_TRAINED_MODEL_SAVE_PATH = "./disdrive/saved_models/disdrive_model.pth" #remove "/disdrive" in the path
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

# ───── MODEL INITIALIZATION ────────────────────────────────────────

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
toggle_state = {"enabled": True}
clients = set()
active_camera = "Front View"
cameras = {
    "Front View": cv2.VideoCapture(0),
    "Side View": cv2.VideoCapture(1)
}

# ───── CAMERA MANAGEMENT ───────────────────────────────────────────

def initialize_cameras():
    for name, cap in cameras.items():
        if not cap.isOpened():
            print(f"❌ Failed to open {name} camera!")
            return False
    return True

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

# ───── DETECTION LOOP ─────────────────────────────────────────────

async def detection_loop():
    """Main video processing loop"""
    try:
        behavior = "Detecting..."  # Initialize with default value
        while True:
            ret, frame = cameras[active_camera].read()
            if not ret:
                print(f"Camera read failed: {active_camera}")
                await asyncio.sleep(1)
                continue

            # Ensure feature extraction and model prediction
            feature = extract_features(frame)
            frame_buffer.append(feature)

            if len(frame_buffer) == 20:
                with torch.no_grad():
                    sequence_tensor = torch.stack(list(frame_buffer)).unsqueeze(0)
                    output = model(sequence_tensor)
                    predicted_label = torch.argmax(output, dim=1).item()
                    behavior = _BEHAVIOR_LABEL.get(predicted_label, "Unknown Behavior")

            # Encode frame for transmission
            _, buffer = cv2.imencode('.jpg', frame)
            latest_data.update({
                "frame": base64.b64encode(buffer).decode('utf-8'),
                "behavior": behavior
            })
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Detection error: {e}")
    finally:
        for cap in cameras.values():
            cap.release()
        print("Released camera resources")

# ───── SEND LOOP ──────────────────────────────────────────

async def send_loop(websocket):
    """Send video frames to client"""
    try:
        while True:
            if latest_data["frame"]:
                await websocket.send(json.dumps({
                    "frame": latest_data["frame"],
                    "behavior": latest_data["behavior"],
                    "camera": active_camera,
                    "toggle": toggle_state["enabled"]
                }))
            await asyncio.sleep(0.033)  # ~30 FPS
    except websockets.ConnectionClosed:
        print("Client disconnected (send loop)")
    except Exception as e:
        print(f"Send error: {e}")

# ───── RECEIVE LOOP ─────────────────────────────────────────────
async def receive_loop(websocket):
    """Handle client messages"""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if "toggle" in data:
                    toggle_state["enabled"] = data["toggle"]
                    print(f"Toggle updated: {toggle_state['enabled']}")
                elif "camera" in data:
                    requested_camera = data["camera"]
                    if requested_camera in ["Front View", "Side View"]:
                        global active_camera
                        active_camera = requested_camera
                        print(f"Camera switched to: {active_camera}")
                        # Reset frame buffer when switching cameras
                        frame_buffer.clear()
            except json.JSONDecodeError:
                print("Invalid JSON received")
    except websockets.ConnectionClosed:
        print("Client disconnected (receive loop)")
    except Exception as e:
        print(f"Receive error: {e}")

async def send_loop(websocket):
    """Send video frames to client"""
    try:
        while True:
            if latest_data["frame"]:
                await websocket.send(json.dumps({
                    "frame": latest_data["frame"],
                    "behavior": latest_data["behavior"],
                    "camera": active_camera,  # Always send the current active camera
                    "toggle": toggle_state["enabled"]
                }))
            await asyncio.sleep(0.033)  # ~30 FPS
    except websockets.ConnectionClosed:
        print("Client disconnected (send loop)")
    except Exception as e:
        print(f"Send error: {e}")

# ───── VIDEO STREAM LOOP ─────────────────────────────────────────────

async def video_stream(websocket, path = '' ):
    """Main connection handler with path parameter"""
    global active_camera
    
    # Parse camera from query params
    query = parse_qs(path.split('?')[1] if '?' in path else {})
    if 'camera' in query and query['camera'][0] in ["Front View", "Side View"]:
        active_camera = query['camera'][0]
    
    clients.add(websocket)
    print(f"Client connected to {active_camera} feed")

    try:
        sender = asyncio.create_task(send_loop(websocket))
        receiver = asyncio.create_task(receive_loop(websocket))
        await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
    finally:
        clients.remove(websocket)
        print("Client disconnected")

# ───── MAIN ENTRY POINT ────────────────────────────────────────────

async def main():
    # Windows-specific setup
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Start services
    detection_task = asyncio.create_task(detection_loop())
    server = await websockets.serve(video_stream, "0.0.0.0", 8765)
    print("✅ Server started on ws://0.0.0.0:8765")

    # Start React frontend
    try:
        print("🚀 Starting React frontend...")
        subprocess.Popen(["npm", "run", "dev"],
                        cwd=_WEBSERVER_PATH, shell=True)
    except Exception as e:
        print(f"⚠️ Failed to start React: {e}")

    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        server.close()
        await server.wait_closed()
        detection_task.cancel()
        try:
            await detection_task
        except asyncio.CancelledError:
            pass
        print("🎉 Clean shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
