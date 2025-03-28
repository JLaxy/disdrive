from datetime import datetime
from backend.database_queries import DatabaseQueries

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class LogManager:
    def __init__(self, database_queries: DatabaseQueries):
        """Handles all log operations"""
        print("Initializing LogManager...")

        self.database_queries = database_queries

        self.has_session = None
        self.session_start = None
        self.current_session_id = None

    def start_session(self):
        """Starts logging session"""
        if self.has_session:
            print("There is already a session running!")
            return

        self.session_start = datetime.now().strftime(_DATETIME_FORMAT)
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

        session_end = datetime.now().strftime(_DATETIME_FORMAT)
        self.has_session = False

        # End session
        print(f"Session ended on {session_end}")
        # Get current date then update session_end and has session

        self.database_queries.log_end_session(
            session_end, self.current_session_id)

        self.current_session_id = None
