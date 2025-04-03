from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, WebSocket
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

        self.delete_old_logs()

        self.connected_clients: List[WebSocket] = []
        self.logs_api = None
        self.configure_logs_api()

    def configure_logs_api(self):
        """Runs all code for configuring API for logs"""
        self.logs_api = FastAPI()

        # Get the absolute path to the database file
        current_dir = Path(__file__).parent
        DB_PATH = (current_dir.parent.parent /
                   "database" / "disdrive_db.db").resolve()

    async def connect_to_api(self, websocket: WebSocket):
        """Allows clients to be connected to API"""
        await websocket.accept()
        self.connected_clients.append(websocket)

    def disconnect_to_api(self, websocket: WebSocket):
        """Disconnects client from server"""
        self.connected_clients.remove(websocket)

    async def broadcast(self, message: str):
        """Sends updates data to clients"""
        for connection in self.connected_clients:
            await connection.send_text(message)

    @self.logs_api.websocket("/ws/logs")
    async def get_all_logs(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM sessions ORDER BY session_start DESC")
                rows = cursor.fetchall()
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
            finally:
                conn.close()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            print(f"Client disconnected")

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

    def delete_old_logs(self):
        """Deletes logs based on retention days setting from the logged_behaviors table."""
        print("Checking for old logs to delete...")

        # Get retention days from settings
        settings = self.database_queries.get_settings()
        retention_days = settings.get('retention_days', 15)  # Default to 15 if not set

        # Calculate cutoff date
        now = datetime.now()
        cutoff_date = now - timedelta(days=retention_days)
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

        print(f"Current system date and time: {now}")
        print(f"Cutoff for {retention_days} days: {cutoff_date_str}")

        # Execute deletion query
        print(f"Executing deletion query for logs older than {retention_days} days...")
        self.database_queries.delete_logs_from_multiple_tables(["logged_behaviors"], cutoff_date_str)
        print("Deletion query executed.")
