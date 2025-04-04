import asyncio
from datetime import datetime
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from backend.database_queries import DatabaseQueries
from typing import List

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_BEHAVIOR_LABEL = {
    "Safe Driving": 0,
    "Texting": 1,
    "Talking using Phone": 2,
    "Drinking": 3,
    "Head Down": 4,
    "Look Behind": 5,
}


class LogManager:
    def __init__(self, database_queries: DatabaseQueries):
        """Handles all log operations"""
        print("Initializing LogManager...")

        self.database_queries = database_queries

        self.has_session = None
        self.session_start = None  # Datetime session has started
        self.behavior = None  # behavior_id of current behavior of driver
        self.behavior_start = None  # Datatime behavior has started
        self.current_session_id = None

        self.connected_clients: List[WebSocket] = []
        self.configure_logs_api()

    def configure_logs_api(self):
        """Runs all code for configuring API for logs"""
        self.logs_api = FastAPI()

        self.logs_api.add_api_websocket_route(
            "/ws/logs", self.get_all_sessions)
        self.logs_api.add_api_websocket_route(
            "/ws/logs/{session_id}", self.get_session_details)

        # Get the absolute path to the database file
        current_dir = Path(__file__).parent
        DB_PATH = (current_dir.parent.parent /
                   "database" / "disdrive_db.db").resolve()

    async def connect_to_api(self, websocket: WebSocket):
        """Allows clients to be connected to API"""
        await websocket.accept()
        self.connected_clients.append(websocket)

    def disconnect_from_api(self, websocket: WebSocket):
        """Disconnects client from server"""
        self.connected_clients.remove(websocket)

    async def broadcast(self, message: str):
        """Sends updates data to clients"""
        for connection in self.connected_clients:
            await connection.send_text(message)

    async def get_all_sessions(self, websocket: WebSocket):
        """Retrieves all sessions saved in database"""
        await self.connect_to_api(websocket)
        try:
            # Fetch from database
            rows = self.database_queries.get_all_sessions()

            # Destructure
            logs = [
                {
                    'session_id': row[0],
                    'session_start': row[1],
                    'session_end': row[2]
                }
                for row in rows
            ]

            await websocket.send_text(json.dumps(logs))
            # Keep connection alive
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.disconnect_from_api(websocket)

    async def get_session_details(self, websocket: WebSocket, session_id: int):
        await self.connect_to_api(websocket)
        print(f"📡 User retrieving details for session_id: {session_id}")
        try:
            while True:
                message = await websocket.receive_text()
                print(f"LOGMANAGER: messaged received: {message}")

                if message != "get_details":
                    continue

                # Get all logged behaviors
                logged_behaviors = self.database_queries.get_all_logged_behaviors(
                    session_id)

                # If session does not exist in database
                if not logged_behaviors:
                    print(
                        f"failed to retrieve session details for session_id: {session_id}")
                    continue

                details = [{"session_id": log[0],
                            "session_start": log[1],
                            "session_end": log[2],
                            "behavior_id": log[3],
                            "behavior": log[4],
                            "behavior_time_start": log[5],
                            "behavior_time_end": log[6]} for log in logged_behaviors]

                print(f"sending: {details}")

                await websocket.send_text(json.dumps(details))

        except WebSocketDisconnect:
            print(f"❌ Client disconnected from session {session_id}")
        except Exception as e:
            print(f"❌ Error in WebSocket connection: {e}")
        finally:
            self.disconnect_from_api(websocket)

    def start_logs_api(self, ip, port):
        """Runs the API server for logs"""
        try:
            config = uvicorn.Config(self.logs_api, host=ip, port=port)
            server = uvicorn.Server(config)

            if asyncio.get_event_loop().is_running():
                asyncio.create_task(server.serve())
            else:
                asyncio.run(server.serve())

        except Exception as e:
            print(f"Failed to run Logs API server!: {e}")

    def get_time_now(self):
        return datetime.now().strftime(_DATETIME_FORMAT)

    def start_session(self):
        """Starts logging session"""
        if self.has_session:
            print("There is already a session running!")
            return

        self.session_start = self.get_time_now()
        self.has_session = True

        print(f"Session started on {self.session_start}")

        self.current_session_id = self.database_queries.log_new_session(
            self.session_start)

        print(f"Session ID: {self.current_session_id}")

    def log_behavior(self, behavior):
        """Logs behavior in current session"""
        if not self.has_session:
            print(
                f"Failed to log behavior {behavior}! there is no active session")
            return

        print(f"Logging behavior: {behavior}")

        # Log behavior

    def end_session(self):
        """Ends current session running"""
        if not self.has_session:
            print("Failed to end session! There is no active session")
            return

        session_end = self.get_time_now()
        self.has_session = False

        # End session
        print(f"Session ended on {session_end}")
        # Get current date then update session_end and has session

        self.database_queries.log_end_session(
            session_end, self.current_session_id)

        self.current_session_id = None

    def new_behavior_started(self, behavior):
        """Records current time new behavior has started"""
        try:
            self.behavior = _BEHAVIOR_LABEL[behavior]
            self.behavior_start = self.get_time_now()
        except Exception as e:
            print(f"ERROR IN LOGMANAGER: {e}")

    def end_behavior(self):
        if self.current_session_id == None:
            print(f"Cannot log behavior {self.behavior}, SessionID not found!")

        if self.behavior == None:
            print(f"Skipping logging behavior...")
            return

        self.database_queries.log_behavior(
            self.behavior, self.current_session_id, self.behavior_start, self.get_time_now())

        self.behavior = None
        self.behavior_start = None
