from datetime import datetime
from backend.database_queries import DatabaseQueries

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
