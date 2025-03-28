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
_SETTINGS = {}


async def main():
    print("Starting Disdrive...")

    # Load Database Handler
    database_query = DatabaseQueries(_PATH_TO_DB)

    # Load Settings from Database
    _SETTINGS = database_query.get_settings()
    print(_SETTINGS["has_ongoing_session"])

    # Load Model
    hybrid_model = DisdriveModel(_SETTINGS)

    # Start Detection Loop
    asyncio.create_task(hybrid_model.detection_loop())

    # Start Websocket
    websocket_service = WebsocketService(hybrid_model, _SETTINGS)
    asyncio.create_task(
        websocket_service.start_disdrive_app_socket("0.0.0.0", 8766))
    asyncio.create_task(
        websocket_service.start_livefeed_socket("0.0.0.0", 8765))

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
        subprocess.Popen(["npm", "run", "dev"],
                         cwd=_WEBSERVER_PATH, shell=True)
    except Exception as e:
        print(f"⚠️ Failed to start React frontend: {e}")


def handle_exit(signum, frame):
    """Handle system signals for graceful shutdown"""
    threading.Thread(target=playsound, args=("./src/end.mp3",), daemon=True).start()
    print("\nReceived exit signal. Shutting down...")
    sys.exit(0)


# Main execution
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.run_forever()
