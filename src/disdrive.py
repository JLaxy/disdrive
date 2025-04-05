from backend.disdrive_model import DisdriveModel
from backend.websocket_service import WebsocketService
from backend.database_queries import DatabaseQueries
import asyncio
import subprocess
import signal
import sys
import pygame
import threading
from pynput import keyboard

_PATH_TO_DB = "./database/disdrive_db.db"
_WEBSERVER_PATH = "./web_server"

def play_sound(file_path: str, channel_id=0):
    """play sound on a specific channel to prevent cutting off other sounds"""
    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    #Create up to 8 channels (different sounds)
    if channel_id >= pygame.mixer.get_num_channels():
        pygame.mixer.set_num_channels(channel_id + 1)

    #Get specific channel
    channel = pygame.mixer.Channel(channel_id)

    #Load and play sound
    sound = pygame.mixer.Sound(file_path)
    channel.play(sound)

    #Only wait for completion if specifically requested
    if channel_id == 0:
        while channel.get_busy():
            pygame.time.wait(100)  #Check less frequently to reduce CPU usage

    

async def handle_system_action(websocket_service : WebsocketService, message_handler, action):
    """Handle system actions"""
    try:
        match action:
            case "start_session":
                if hasattr(websocket_service, 'message_handler'):
                    await websocket_service.message_handler.start_session(None, None)
            case "stop_session":
                if hasattr(websocket_service, 'message_handler'):
                    await websocket_service.message_handler.stop_session(None, None)
            case "shutdown_system":
                print("Shutting down system...")
                if hasattr(websocket_service, 'message_handler'):
                    await websocket_service.message_handler.shutdown_system(None, websocket_service)
                sys.exit(0)
    except Exception as e:
        print(f"Error in handle_system_action: {e}")
    finally:
        await asyncio.create_task(websocket_service.broadcast_settings())

def on_key_press(key, websocket_service, hybrid_model):
    """Handle keyboard events"""
    try:
        if hasattr(key, 'char'):
            # Create new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            match key.char.upper():
                case 'A':
                    print("Start key pressed")
                    loop.run_until_complete(handle_system_action(websocket_service, hybrid_model, "start_session"))
                case 'B':
                    print("Stop key pressed")
                    loop.run_until_complete(handle_system_action(websocket_service, hybrid_model, "stop_session"))
                case 'C':
                    print("Shutdown key pressed")
                    loop.run_until_complete(handle_system_action(websocket_service, hybrid_model, "shutdown_system"))
            
            loop.close()
    except Exception as e:
        print(f"Error handling key press: {e}")

async def main():
    print("Starting Disdrive...")

    # Load Database Handler
    database_query = DatabaseQueries(_PATH_TO_DB)

    # Load Model
    hybrid_model = DisdriveModel(database_query)
    
    # Create WebSocket Service
    websocket_service = WebsocketService(
        hybrid_model, database_query)

    # Create tasks for detection and websockets
    detection_task = asyncio.create_task(hybrid_model.detection_loop())

    hybrid_model.log_manager.start_logs_api("0.0.0.0", 8767)

    # Use asyncio.create_task with explicit server methods
    disdrive_socket_task = asyncio.create_task(
        websocket_service.start_disdrive_app_socket("0.0.0.0", 8766)
    )
    livefeed_socket_task = asyncio.create_task(
        websocket_service.start_livefeed_socket("0.0.0.0", 8765)
    )

    # Start Frontend
    frontend_process = start_frontend()

    #Startup sounds
    threading.Thread(target=play_sound, args=("src/assets/startup.mp3", 0), daemon=True).start()

    # Start keyboard listener with both services
    keyboard_listener = keyboard.Listener(
        on_press=lambda key: on_key_press(key, websocket_service, hybrid_model)
    )
    keyboard_listener.start()

    try:
        # Wait for all tasks
        await asyncio.gather(
            detection_task,
            disdrive_socket_task,
            livefeed_socket_task
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        keyboard_listener.stop()  # Stop keyboard listener
        # Cleanup
        if frontend_process:
            frontend_process.terminate()
        websocket_service.stop_servers()


def start_frontend():
    """Starts React Frontend"""
    try:
        print("🚀 Starting React frontend...")
        return subprocess.Popen(
            "npm run dev",
            cwd=_WEBSERVER_PATH,
            shell=True
        )
    except Exception as e:
        print(f"⚠️ Failed to start React frontend: {e}")
        return None


def handle_exit(signum, frame):
    """Handle system signals for graceful shutdown"""
    print("\nReceived exit signal. Shutting down...")
    sys.exit(0)


# Main execution
if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Set up event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down...")
    finally:
        loop.close()
