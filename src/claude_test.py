import cv2
import torch
import numpy as np
from frame_sequences.hybrid_model import HybridModel
from collections import deque
import threading
import time
from queue import Queue
from PIL import Image
_TRAINED_MODEL_SAVE_PATH = "./saved_models/disdrive_model.pth"
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

# Global variables
frame_buffer = deque(maxlen=20)
processing_active = True
prediction_queue = Queue(maxsize=5)
display_queue = Queue(maxsize=10)
actual_fps = 0
processing_fps = 0


def load_model() -> HybridModel:
    """Loads and initializes model"""
    CLIP_LSTM = HybridModel()
    CLIP_LSTM.load_state_dict(torch.load(
        _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
    CLIP_LSTM.to(_DEVICE)
    CLIP_LSTM.eval()
    return CLIP_LSTM


def extract_features(model, frame):
    """Preprocesses frame then extracts feature from frame"""
    frame = cv2.resize(frame, (224, 224))
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = Image.fromarray(processed_frame)
    processed_frame = model.preprocessor(
        processed_frame).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        features = model.clip_model.encode_image(
            processed_frame).squeeze(0).to(torch.float32)

    return features


def initialize_camera(camera_id=0):
    """Initialize camera with optimal settings for speed"""
    # Try different backends in order of preference
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        try:
            # Try to open camera with current backend
            camera = cv2.VideoCapture(camera_id, backend)
            if not camera.isOpened():
                continue

            # Configure camera for speed
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)

            # Reduce internal buffer size to prevent lag
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify camera is working by reading a test frame
            ret, test_frame = camera.read()
            if ret:
                print(f"Successfully opened camera with backend: {backend}")
                return camera
            else:
                camera.release()
        except Exception as e:
            print(f"Error with backend {backend}: {e}")

    # If all backends failed, try one last time with default settings
    print("Falling back to default camera settings")
    camera = cv2.VideoCapture(camera_id)
    if camera.isOpened():
        return camera

    # If everything failed
    raise RuntimeError("Failed to open camera with any backend")


def process_frames(model):
    """Process frames in a separate thread"""
    global frame_buffer, processing_active, processing_fps

    frames_processed = 0
    process_timer = time.time()

    while processing_active:
        if len(frame_buffer) >= 20:
            try:
                with torch.no_grad():
                    # Get frames from buffer as a batch
                    sequence_tensor = torch.stack(list(frame_buffer))
                    sequence_tensor = sequence_tensor.unsqueeze(0)

                    # Make prediction
                    output = model(sequence_tensor)
                    prediction = torch.argmax(output, dim=1).item()

                    # Add to prediction queue - don't block if queue is full
                    try:
                        prediction_queue.put_nowait(prediction)
                    except:
                        pass  # Skip if queue is full

                    # Update processing FPS
                    frames_processed += 1
                    if time.time() - process_timer >= 1.0:
                        processing_fps = frames_processed
                        frames_processed = 0
                        process_timer = time.time()
            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
        else:
            # Not enough frames, wait a bit
            time.sleep(0.05)


def capture_frames(model, camera):
    """Capture frames with optimized approach"""
    global frame_buffer, processing_active, actual_fps

    frames_captured = 0
    capture_timer = time.time()

    # Main capture loop
    while processing_active:
        try:
            # Read multiple frames to flush any buffered frames
            for _ in range(2):  # Flush buffer by reading extra frames
                camera.grab()  # Just grab frame, don't decode

            # Read the actual frame we want to process
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame!")
                # Try to reinitialize camera
                camera.release()
                time.sleep(0.5)
                camera = initialize_camera()
                continue

            # Add frame to display queue without blocking
            try:
                display_queue.put_nowait((frame.copy(), time.time()))
            except:
                pass  # Skip if queue is full

            # Extract features - only do this if we're not overwhelmed
            if len(frame_buffer) < 20:
                feature = extract_features(model, frame)
                frame_buffer.append(feature)

            # Update capture FPS counter
            frames_captured += 1
            if time.time() - capture_timer >= 1.0:
                actual_fps = frames_captured
                frames_captured = 0
                capture_timer = time.time()

        except Exception as e:
            print(f"Error in capture thread: {e}")
            time.sleep(0.1)  # Prevent tight loop on error


def display_loop():
    """Display frames with prediction overlay"""
    global processing_active, actual_fps, processing_fps

    last_prediction = None
    display_fps = 0
    frame_display_count = 0
    display_timer = time.time()

    while processing_active:
        try:
            # Get the latest prediction if available
            if not prediction_queue.empty():
                last_prediction = prediction_queue.get()

            # Try to get a frame with timeout to prevent blocking forever
            try:
                frame, timestamp = display_queue.get(timeout=0.1)

                # Calculate frame age
                frame_age = (time.time() - timestamp) * 1000  # in milliseconds

                # Add information overlays
                cv2.putText(
                    frame, f"Capture FPS: {actual_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

                cv2.putText(
                    frame, f"Processing FPS: {processing_fps}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

                cv2.putText(
                    frame, f"Display FPS: {display_fps}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

                cv2.putText(
                    frame, f"Frame Latency: {frame_age:.1f}ms", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

                # Show behavior prediction
                if last_prediction is not None:
                    behavior = _BEHAVIOR_LABEL[last_prediction]
                    cv2.putText(
                        frame, f"Behavior: {behavior}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,
                                                        0, 255), 2, cv2.LINE_AA
                    )

                # Display the frame
                cv2.imshow("Distracted Driving Detection", frame)

                # Count displayed frames for display FPS
                frame_display_count += 1
                if time.time() - display_timer >= 1.0:
                    display_fps = frame_display_count
                    frame_display_count = 0
                    display_timer = time.time()

                # Check for quit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    processing_active = False
                    break
            except:
                # No frames available, just check for quit
                key = cv2.waitKey(100)
                if key == ord('q'):
                    processing_active = False
                    break

        except Exception as e:
            print(f"Error in display thread: {e}")
            time.sleep(0.1)  # Prevent tight loop on error


def main():
    global processing_active

    print("Initializing camera...")
    try:
        camera = initialize_camera()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return

    print(f"Loading model on {_DEVICE}...")
    model = load_model()
    print("Model loaded successfully!")

    # Create and start threads
    print("Starting processing threads...")

    # Processing thread
    process_thread = threading.Thread(
        target=process_frames, args=(model,), daemon=True)
    process_thread.start()

    # Capture thread - give it higher priority if possible
    capture_thread = threading.Thread(
        target=capture_frames, args=(model, camera), daemon=True)
    try:
        # Try to set thread priority higher (works on some systems)
        capture_thread.setDaemon(True)
    except:
        pass
    capture_thread.start()

    # Display in main thread
    print("Starting display loop...")
    try:
        display_loop()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        processing_active = False
        print("Waiting for threads to finish...")
        process_thread.join(timeout=1.0)
        capture_thread.join(timeout=1.0)
        camera.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")


if __name__ == "__main__":
    main()
