from backend.disdrive_model import DisdriveModel
from backend.websocket_service import WebsocketService
from backend.database_queries import DatabaseQueries
import asyncio
import subprocess

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
    hybrid_model = DisdriveModel(
        _SETTINGS["camera_id"], _SETTINGS["has_ongoing_session"])

    # Start Detection Loop
    asyncio.create_task(hybrid_model.detection_loop())

    # Start Websocket
    websocket_service = WebsocketService(hybrid_model, _SETTINGS)
    asyncio.create_task(
        websocket_service.start_disdrive_app_socket("0.0.0.0", 8766))
    asyncio.create_task(
        websocket_service.start_livefeed_socket("0.0.0.0", 8765))

    # Start Frontend
    start_frontend()


def start_frontend():
    """Starts React Frontend"""
    try:
        print("🚀 Starting React frontend...")
        subprocess.Popen(["npm", "run", "dev"],
                         cwd=_WEBSERVER_PATH, shell=True)
    except Exception as e:
        print(f"⚠️ Failed to start React frontend: {e}")


# Main function
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.run_forever()
