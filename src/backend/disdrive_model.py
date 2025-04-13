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
import concurrent.futures
import multiprocessing
import time
import pygame

# Fix device selection - CORRECTED
_TRAINED_MODEL_SAVE_PATH = "./saved_models/final_model.pth"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BEHAVIOR_LABEL = {
    0: "Safe Driving",
    1: "Texting",
    2: "Talking using Phone",
    3: "Drinking",
    4: "Head Down",
    5: "Look Behind",
}

# Optimized configuration for ROCM/APU performance
_FRAME_SKIP = 1  # Process every frame for smoother detection
_FRAME_WIDTH = 224  # Standard size for model input
_FRAME_HEIGHT = 224  # Standard size for model input
_BUFFER_SIZE = 20  # Frames to analyze
_SLIDING_WINDOW_STEP = 3  # Slide window by this many frames
_MAX_WORKERS = max(4, multiprocessing.cpu_count() - 2)  # Use more CPU cores
_FEATURE_QUEUE_SIZE = 20  # Larger queue
_FRAME_QUEUE_SIZE = 20  # Larger queue sizesize

# Alert settings
_ALERT_INTERVAL = 1  # Play sound every second
_ALERT_SOUND_PATH = "./src/assets/alert.mp3"

# Initialize pygame mixer once
pygame.mixer.init()


def play_sound(file_path: str, channel_id=0):
    """play sound on a specific channel to prevent cutting off other sounds"""
    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # Create up to 8 channels (different sounds)
    if channel_id >= pygame.mixer.get_num_channels():
        pygame.mixer.set_num_channels(channel_id + 1)

    # Get specific channel
    channel = pygame.mixer.Channel(channel_id)

    # Load and play sound
    sound = pygame.mixer.Sound(file_path)
    channel.play(sound)

    # Only wait for completion if specifically requested
    if channel_id == 0:
        while channel.get_busy():
            pygame.time.wait(100)  # Check less frequently to reduce CPU usage


class DisdriveModel:
    """Handles all functionalities related to the Machine Learning Model"""

    def __init__(self, database_queries: DatabaseQueries):
        self._initialize_model()

        self.database_queries = database_queries
        self.log_manager = LogManager(self.database_queries)
        self.to_log = None
        self.update_session_status()

        self._initialize_cameras()

        self._configure_threads()

        # Alert tracking variables
        self.last_alert_time = 0
        self.alert_playing = False
        self.unsafe_behavior_detected = False

        saved_camera = self.get_selected_camera_saved()
        self._detection_loop_task = None
        self._feature_extraction_tasks = []

        # If no saved camera or saved camera is available
        if saved_camera == None or (saved_camera not in self.available_cameras):
            print(
                f"DISDRIVE_MODEL: Saved camera with index {saved_camera} not available! opening nearest available camera...")
            # Open first available camera
            self.open_camera(self.available_cameras[0])
        else:
            self.open_camera(saved_camera)

    def _initialize_model(self):
        """All functions to initialize model"""
        print(f"Using device: {_DEVICE} with {_MAX_WORKERS} workers")
        self.model = HybridModel()
        self.model.load_state_dict(torch.load(
            _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
        self.model.to(_DEVICE)
        self.model.eval()

        # Enable optimizations for ROCM
        if _DEVICE == "cuda":
            # These settings help ROCM performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')

        # Create larger thread pool for feature extraction
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_WORKERS)

        # Preload and cache preprocessor transforms
        if hasattr(self.model, 'clip_model') and hasattr(self.model, 'preprocessor'):
            print("Warming up CLIP model...")
            # Warmup with batch processing to initialize ROCM kernels
            dummy_batch = torch.zeros(4, 3, 224, 224).to(_DEVICE)
            with torch.no_grad():
                self.model.clip_model.encode_image(dummy_batch)
                dummy_sequence = torch.zeros(2, _BUFFER_SIZE, 512).to(_DEVICE)
                self.model(dummy_sequence)

        # Use deque with set buffer size
        self.frame_buffer = deque(maxlen=_BUFFER_SIZE)
        self.latest_detection_data = {
            "frame": None, "behavior": "Detecting...", "fps": "0.0"}
        self.probabilities = [0] * len(_BEHAVIOR_LABEL)

    def _initialize_cameras(self):
        """Configures camera"""
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
        self.window_slide_counter = 0

    def _configure_threads(self):
        """Configures threads to be ran"""
        # Create a feature cache to avoid repetitive computations
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.MAX_CACHE_SIZE = 500  # Increased cache size

        # Threading for feature extraction - use larger queues
        self.feature_queue = asyncio.Queue(maxsize=_FEATURE_QUEUE_SIZE)
        self.frame_queue = asyncio.Queue(maxsize=_FRAME_QUEUE_SIZE)

        # Add semaphore to control concurrent feature extractions
        self.feature_semaphore = asyncio.Semaphore(_MAX_WORKERS)

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
        Open a specific camera by its index with optimized settings.

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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Attempt to set higher FPS - many cameras support more than 30
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Get actual camera properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(
            f"Camera configured with: {actual_width}x{actual_height} @ {actual_fps}fps")

        # Set buffer size to 2 for better throughput but still low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.current_camera_index = camera_index

        # If cannot open camera, print error
        if not self.cap.isOpened():
            print(f"❌ Unable to open camera with index {camera_index}!")
            self.current_camera_index = None
        else:
            # Update opened camera
            self.database_queries.update_setting("camera_id", camera_index)

    def update_session_status(self):
        """Updates the session and is_logging status"""
        settings = self.database_queries.get_settings()

        self.has_ongoing_session = bool(settings["has_ongoing_session"])
        self.to_log = bool(settings["is_logging"])

    def extract_features(self, frame):
        """Extracts features of retrieved frame from camera with optimized processing"""
        # Generate a simple hash for the frame to check cache
        # Use a more efficient hashing method
        # Hash just part of the frame for speed
        frame_hash = hash(frame.tobytes()[:1000])

        # Check if we've already computed features for this frame
        if frame_hash in self.feature_cache:
            self.cache_hits += 1
            if self.cache_hits % 100 == 0:
                print(
                    f"Feature cache hits: {self.cache_hits}, misses: {self.cache_misses}")
            return self.feature_cache[frame_hash]

        self.cache_misses += 1

        # Convert to RGB for PIL
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = Image.fromarray(processed_frame)

        # Use the CLIP preprocessor
        processed_frame = self.model.preprocessor(
            processed_frame).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            features = self.model.clip_model.encode_image(
                processed_frame).squeeze(0).to(torch.float32)

        # Store in cache - use LRU approach
        if len(self.feature_cache) >= self.MAX_CACHE_SIZE:
            # Remove oldest key if cache is full
            self.feature_cache.pop(next(iter(self.feature_cache)))

        self.feature_cache[frame_hash] = features
        return features

    async def feature_extraction_worker(self, worker_id):
        """Worker to extract features asynchronously"""
        try:
            print(f"Feature extraction worker {worker_id} started")
            while True:
                if asyncio.current_task().cancelled():
                    print(f"Feature extraction worker {worker_id} cancelled")
                    break

                # Get frame from queue with short timeout
                try:
                    frame = await asyncio.wait_for(self.frame_queue.get(), 0.5)
                except asyncio.TimeoutError:
                    continue

                # Use ThreadPoolExecutor for CPU-bound operations
                async with self.feature_semaphore:
                    loop = asyncio.get_event_loop()
                    feature = await loop.run_in_executor(self.executor, self.extract_features, frame)

                # Put feature in queue for main loop
                try:
                    await asyncio.wait_for(self.feature_queue.put(feature), 0.1)
                except asyncio.TimeoutError:
                    # If feature queue is full, we can skip this frame
                    pass

                # Mark task as done
                self.frame_queue.task_done()

        except asyncio.CancelledError:
            print(f"Feature extraction worker {worker_id} was cancelled")
        except Exception as e:
            print(f"Error in feature extraction worker {worker_id}: {e}")

    async def process_frame_buffer(self):
        """Process the current frame buffer and predict behavior"""
        try:
            # Only process if we have enough frames
            if len(self.frame_buffer) < _BUFFER_SIZE:
                return "Detecting..."  # Not enough frames yet

            with torch.no_grad():
                # Create tensor from buffer
                buffer_list = list(self.frame_buffer)
                sequence_tensor = torch.stack(
                    buffer_list).unsqueeze(0).to(_DEVICE)

                # Run model inference
                output = self.model(sequence_tensor)

                # Get probabilities using softmax
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                probabilities = probabilities.cpu().numpy()

                # Get predicted class
                predicted_class = torch.argmax(output, dim=1).item()
                behavior = _BEHAVIOR_LABEL[predicted_class]

                return behavior, probabilities
        except Exception as e:
            print(f"Error processing frame buffer: {e}")
            return "Error"

    def play_continuous_alert(self, behavior):
        """Play alert sound continuosly when behavior is unsafe"""
        current_time = time.time()

        # Check if behavior is unsafe
        unsafe_behavior = behavior != "Safe Driving" and behavior != "Detecting..." and behavior != "Error" and behavior != "Detection Paused"

        # Update unsafe behavior status
        self.unsafe_behavior_detected = unsafe_behavior

        # Play alert at regular intervals if unsafe behavior continues
        if unsafe_behavior and (current_time - self.last_alert_time >= _ALERT_INTERVAL):
            self.last_alert_time = current_time
            threading.Thread(target=play_sound, args=(
                _ALERT_SOUND_PATH, 1), daemon=True).start()
            print("Alert sound triggered for:", behavior)

    async def detection_loop(self):
        """Responsible for detecting behavior of driver with optimized processing"""

        print("Starting Detection...")
        self.fps_start_time = asyncio.get_event_loop().time()
        self.frame_count = 0
        self.frame_skip_counter = 0
        self.window_slide_counter = 0

        # Start multiple feature extraction workers
        self._feature_extraction_tasks = []
        for i in range(_MAX_WORKERS):
            worker = asyncio.create_task(self.feature_extraction_worker(i))
            self._feature_extraction_tasks.append(worker)

        try:
            while True:
                # Check if task will be cancelled
                if asyncio.current_task().cancelled():
                    print("Detection loop cancelled")
                    break

                # Read frame with timeout to prevent blocking
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    await asyncio.sleep(0.001)  # Reduced sleep time
                    continue

                # Count each successfully captured frame
                self.frame_count += 1

                # Calculate and print FPS every second
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - self.fps_start_time

                if elapsed_time >= 1.0:  # Update FPS more frequently
                    self.current_fps = self.frame_count / elapsed_time
                    # print(
                    #     f"Current FPS: {self.current_fps:.2f} | Processed frames: {self.processed_frames}")
                    self.frame_count = 0
                    self.processed_frames = 0
                    self.fps_start_time = current_time

                # # Encode frame for UI with reduced resolution for faster encoding
                # display_frame = cv2.resize(frame, (320, 240))
                # _, buffer = cv2.imencode('.jpg', display_frame, [
                #                          cv2.IMWRITE_JPEG_QUALITY, 70])   Z

                # Draw probabilities on frame
                y_offset = 30
                for i, prob in enumerate(self.probabilities):
                    text = f"{_BEHAVIOR_LABEL[i]}: {prob:.2%}"
                    cv2.putText(frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20

                _, buffer = cv2.imencode('.jpg', frame, [
                                         cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                self.latest_detection_data["frame"] = frame_bytes
                self.latest_detection_data["fps"] = f"{self.current_fps:.1f}"

                # If to not detect, skip detection
                if not self.has_ongoing_session:
                    self.latest_detection_data["behavior"] = "Detection Paused"

                    if self.log_manager.has_session:
                        self.log_manager.end_session()

                    # Reduced sleep time
                    await asyncio.sleep(0.001)
                    continue

                # If still not started, then start
                if not self.log_manager.has_session:
                    self.log_manager.start_session()

                # Process frame for detection
                if self.frame_skip_counter % _FRAME_SKIP == 0:
                    # Queue the frame for feature extraction
                    if not self.frame_queue.full():
                        # Use try/except with timeout to avoid blocking
                        try:
                            await asyncio.wait_for(self.frame_queue.put(frame.copy()), 0.01)
                            self.processed_frames += 1
                        except asyncio.TimeoutError:
                            pass  # Skip this frame if queue is full

                    # Try to get features from the queue - more aggressive
                    # Process up to 3 features per cycle
                    for _ in range(min(3, self.feature_queue.qsize())):
                        try:
                            feature = await asyncio.wait_for(self.feature_queue.get(), 0.001)
                            self.frame_buffer.append(feature)
                            self.feature_queue.task_done()
                            self.window_slide_counter += 1
                        except (asyncio.QueueEmpty, asyncio.TimeoutError):
                            break

                self.frame_skip_counter += 1

                # Get current behavior
                behavior = self.latest_detection_data.get(
                    "behavior", "Detecting...")

                # Predict whenever buffer is full or we've added enough new frames
                if (len(self.frame_buffer) >= _BUFFER_SIZE and
                        self.window_slide_counter >= _SLIDING_WINDOW_STEP):
                    self.window_slide_counter = 0  # Reset counter
                    new_behavior, probabilities = await self.process_frame_buffer()

                    # Update behavior if changed
                    if new_behavior != "Detecting..." and new_behavior != behavior:
                        behavior = new_behavior

                        # Add probabilities to display on frame
                        if probabilities is not None:
                            self.probabilities = probabilities

                        # Log behavior change
                        if self.to_log:
                            self.log_manager.end_behavior()
                            self.log_manager.new_behavior_started(
                                behavior, buffer.tobytes())

                # Update shared state for all clients to access
                self.latest_detection_data["behavior"] = behavior

                # Play continuous alert if behavior is unsafe
                self.play_continuous_alert(behavior)

                # Use a very minimal sleep to yield control
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            print("Detection loop was cancelled")
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            # Cancel feature extraction workers
            for worker in self._feature_extraction_tasks:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

            # Ensure camera is released
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            # Shutdown executor
            self.restart_feature_extraction()

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

            # Set camera properties for better performance - try higher FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 60)  # Try for 60fps
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            # Validate new camera capture
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}")

            self.frame_buffer.clear()
            self.feature_cache.clear()  # Clear feature cache on camera change

            # Reset FPS tracking
            self.frame_count = 0
            self.fps_start_time = None
            self.current_fps = 0
            self.processed_frames = 0
            self.frame_skip_counter = 0
            self.window_slide_counter = 0

            # Update current camera index
            self.current_camera_index = camera_id

            # Restart loop
            self._detection_loop_task = asyncio.create_task(
                self.detection_loop())

            print(f"Successfully changed to camera {camera_id}")

        except Exception as e:
            print(f"Error changing camera: {e}")

    def restart_feature_extraction(self):
        if self.executor:
            print("Shutting down executor pool...")
            self.executor.shutdown(wait=False)  # Less blocking shutdown
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_MAX_WORKERS)
            print(f"Executor pool restarted with {_MAX_WORKERS} workers")

    # Model path update depending on camera view
    async def update_model_path(self, model_path: str):
        """
        Update the model path and reload the model.
        Args:
            model_path (str): The new path to the model file.
        """
        try:
            print(f"Updating model path to: {model_path}")
            global _TRAINED_MODEL_SAVE_PATH
            _TRAINED_MODEL_SAVE_PATH = model_path

            # Reload the model with the new path
            self.model.load_state_dict(torch.load(
                _TRAINED_MODEL_SAVE_PATH, map_location=_DEVICE))
            self.model.to(_DEVICE)
            self.model.eval()
            print("Model reloaded successfully.")
        except Exception as e:
            print(f"Error updating model path: {e}")
