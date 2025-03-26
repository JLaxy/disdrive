from backend.disdrive_model import DisdriveModel
from backend.websocket_service import WebsocketService
from backend.database_queries import DatabaseQueries
import asyncio

_PATH_TO_DB = "./database/disdrive_db.db"


async def main():
    print("Starting Disdrive...")

    # Load Database Handler
    database_query = DatabaseQueries(_PATH_TO_DB)
    # Load Settings from Database
    # Load Model
    hybrid_model = DisdriveModel()
    # Start Detection Loop
    # Start Websocket
    websocket_service = WebsocketService()
    # Start Frontend


# Main function
if __name__ == "__main__":
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # loop.run_until_complete(main())
    # loop.run_forever()

    print(DatabaseQueries(_PATH_TO_DB).get_settings())
