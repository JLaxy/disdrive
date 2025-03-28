from backend.disdrive_model import DisdriveModel
from backend.websocket_service import WebsocketService
from backend.database_queries import DatabaseQueries
import asyncio
import subprocess
import signal
import sys
from playsound import playsound
import threading

_PATH_TO_DB = "./database/disdrive_db.db"
_WEBSERVER_PATH = "./web_server"


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

    # Use asyncio.create_task with explicit server methods
    disdrive_socket_task = asyncio.create_task(
        websocket_service.start_disdrive_app_socket("0.0.0.0", 8766)
    )
    livefeed_socket_task = asyncio.create_task(
        websocket_service.start_livefeed_socket("0.0.0.0", 8765)
    )

    # Start Frontend
    frontend_process = start_frontend()

    #Startup sound
    threading.Thread(target=playsound, args=("./src/startup.mp3",), daemon=True).start()

    try:
        # Wait for all tasks
        await asyncio.gather(
            detection_task,
            disdrive_socket_task,
            livefeed_socket_task
        )
    except asyncio.CancelledError:
        threading.Thread(target=playsound, args=("./src/error detected.mp3",), daemon=True).start()
        print("Tasks were cancelled")
    except Exception as e:
        threading.Thread(target=playsound, args=("./src/error detected.mp3",), daemon=True).start()
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if frontend_process:
            frontend_process.terminate()
        websocket_service.stop_servers()


def start_frontend():
    """Starts React Frontend"""
    try:
        print("🚀 Starting React frontend...")
        return subprocess.Popen(
            ["npm", "run", "dev"],
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
