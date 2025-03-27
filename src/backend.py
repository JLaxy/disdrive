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
import time
from playsound import playsound
import threading

# ───── CONFIGURATION ───────────────────────────────────────────────
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
# Shared boolean toggle controlled by frontend
toggle_state = {"enabled": True}
clients = set()  # Store connected clients

# ───── CAMERA SETUP ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to open camera!")
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

# ───── ALERT SOUND ─────────────────────────────────────────────────
def play_alert_sound():
    threading.Thread(target=playsound, args=("./src/alert.mp3",), daemon=True).start()

# ───── DETECTION LOOP (RUNS IN BACKGROUND) ─────────────────────────
async def detection_loop():
    global last_alert_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Extract features using CLIP encoder
        feature = extract_features(frame)
        frame_buffer.append(feature)

        behavior = "Detecting..."
        if len(frame_buffer) == 20:
            with torch.no_grad():
                sequence_tensor = torch.stack(list(frame_buffer)).unsqueeze(0)
                output = model(sequence_tensor)
                output = torch.argmax(output, dim=1).item()
                behavior = _BEHAVIOR_LABEL[output]
                
                # Play alert sound if enabled
                if 'last_alert_time' not in globals():
                    last_alert_time = 0
                current_time = time.time()
                print(current_time)
                if behavior != "Safe Driving" and (current_time - last_alert_time) >= 1:
                    play_alert_sound()
                    last_alert_time = current_time

        # Encode frame to base64 for transmission
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        # Update shared state for all clients to access
        latest_data["frame"] = frame_bytes
        latest_data["behavior"] = behavior

        await asyncio.sleep(0.01)  # Prevent busy-waiting

# ───── SEND LOOP (FRAME + BEHAVIOR TO FRONTEND) ────────────────────


async def send_loop(websocket):
    try:
        while True:
            if latest_data["frame"] is not None:
                await websocket.send(json.dumps({
                    "frame": latest_data["frame"],
                    "behavior": latest_data["behavior"],
                    "toggle": toggle_state["enabled"]
                }))
            await asyncio.sleep(0.01)
    except websockets.ConnectionClosed:
        print("📤 send_loop: Client disconnected")
    except Exception as e:
        print(f"⚠️ send_loop error: {e}")

# ───── RECEIVE LOOP (HANDLE TOGGLE FROM FRONTEND) ──────────────────


async def receive_loop(websocket):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if "toggle" in data:
                    toggle_state["enabled"] = data["toggle"]
                    print(f"🔁 Toggled alert: {toggle_state['enabled']}")
            except json.JSONDecodeError:
                print("⚠️ Received invalid JSON from frontend.")
    except websockets.ConnectionClosed:
        print("📥 receive_loop: Client disconnected")
    except Exception as e:
        print(f"⚠️ receive_loop error: {e}")

# ───── COMBINED CLIENT HANDLER ─────────────────────────────────────


async def video_stream(websocket):
    clients.add(websocket)
    print("📡 Client connected")

    # Create concurrent send and receive tasks for each client
    sender_task = asyncio.create_task(send_loop(websocket))
    receiver_task = asyncio.create_task(receive_loop(websocket))

    # Wait until one task finishes (disconnect or error)
    done, pending = await asyncio.wait(
        [sender_task, receiver_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel any remaining tasks and clean up
    for task in pending:
        task.cancel()

    clients.remove(websocket)
    print("🔌 Client disconnected")

# ───── MAIN ENTRY POINT ────────────────────────────────────────────


async def main():
    # Start model detection in background
    asyncio.create_task(detection_loop())

    # Start WebSocket server
    async with websockets.serve(video_stream, "0.0.0.0", 8765):
        print("✅ WebSocket server started on ws://0.0.0.0:8765")

        # Automatically start React frontend
        try:
            print("🚀 Starting React frontend...")
            subprocess.Popen(["npm", "run", "dev"],
                             cwd=_WEBSERVER_PATH, shell=True)
        except Exception as e:
            print(f"⚠️ Failed to start React frontend: {e}")

        await asyncio.Future()  # Keep server running forever

# ───── RUN APP ─────────────────────────────────────────────────────
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.run_forever()
