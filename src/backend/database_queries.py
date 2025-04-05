from typing import Any
from backend.database_manager import DatabaseManager
from datetime import datetime


class DatabaseQueries:
    """Class handling all database queries to Disdrive Database"""

    def __init__(self, path_to_db):
        print("Initializing DatabaseQueries...")
        self.db_manager: DatabaseManager = DatabaseManager(path_to_db)
        self.auto_start_session()

    def get_settings(self):
        """Returns saved settings in dictionary form from database"""
        result = self.db_manager.fetch_one(
            "SELECT * FROM disdrive_settings")

        if result is None:
            # Default settings if no result found
            return {
                "is_logging": True,
                "camera_id": None,
                "has_ongoing_session": True,
                "retention_days": 15
            }

        # Safely handle the case where retention_days column might not exist yet
        settings = {
            "is_logging": bool(result[0]),
            "camera_id": result[1],
            "has_ongoing_session": bool(result[2]),
            "retention_days": 15  # Default value
        }

        # Try to get retention_days if it exists
        try:
            if len(result) > 3 and result[3] is not None:
                settings["retention_days"] = result[3]
        except IndexError:
            print("retention_days column not found, using default value of 15")

        return settings

    def auto_start_session(self):
        """Auto start session on app start"""
        self.update_setting('has_ongoing_session', True)

    def log_new_session(self, date: str):
        """Logs new session on database; returns session ID"""
        print(f"logging date: {date} type: {type(date)}")

        query = "INSERT INTO sessions (session_start) VALUES (?)"
        self.db_manager.insert(query, (date,))

        return self.db_manager.fetch_one("SELECT * FROM sessions WHERE session_start = :session_start", {"session_start": f"{date}"})[0]

    def log_end_session(self, end: str, session_id: str):
        """Ends existing session on database"""
        self.db_manager.update(
            "UPDATE sessions SET session_end = :session_end WHERE session_id = :session_id", {"session_end": f"{end}", "session_id": session_id})

    def update_setting(self, key: str, value: Any):
        """
        Update a single setting in the database

        Args:
            key: Name of the setting to update
            value: New value for the setting
        """
        # Mapping of settings to their database column names
        setting_map = {
            'is_logging': 'is_logging',
            'camera_id': 'camera_id',
            'has_ongoing_session': 'has_ongoing_session',
            'retention_days': 'retention_days'
        }

        # Validate the setting key
        if key not in setting_map:
            raise ValueError(f"Invalid setting key: {key}")

        # Prepare the query
        query = f"UPDATE disdrive_settings SET {setting_map[key]} = ?"

        # Execute the update
        self.db_manager.update(query, (value,))

    def log_behavior(self, behavior_id: int, session_id: int, behavior_time_start: datetime, behavior_time_end: datetime, snapshot):
        query = f"INSERT INTO logged_behaviors (behavior_id, session_id, behavior_time_start, behavior_time_end, snapshot) VALUES (?, ?, ?, ?, ?)"
        self.db_manager.insert(
            query, (behavior_id, session_id, behavior_time_start, behavior_time_end, snapshot))

    def get_all_sessions(self):
        """Retrieves all sessions in database"""
        query = "SELECT * FROM sessions ORDER BY session_start DESC"
        return self.db_manager.fetch_all(query)

    def get_session_details(self, session_id: int):
        """Retrieves session details from database"""
        query = f"SELECT * FROM sessions WHERE session_id = ?"
        return self.db_manager.fetch_one(query, (session_id,))

    def get_all_logged_behaviors(self, session_id: int):
        """Retrieves all logged behaviors of specific session from database"""
        query = f"SELECT s.session_id, s.session_start, s.session_end, b.behavior_id, behavior, behavior_time_start, behavior_time_end, snapshot FROM logged_behaviors lb JOIN behaviors b ON lb.behavior_id = b.behavior_id JOIN sessions s ON lb.session_id = s.session_id WHERE s.session_id = ?"
        return self.db_manager.fetch_all(query, (session_id,))

    def delete_logs_from_multiple_tables(self, table_names: list[str], cutoff_date: str):
        for table_name in table_names:
            # Compare the full timestamp directly
            query = f"DELETE FROM {table_name} WHERE behavior_time_start < ?"
            try:
                print(f"Attempting to delete logs from table: {table_name}")
                print(f"Cutoff date: {cutoff_date}")
                self.db_manager.delete(query, (cutoff_date,))
                print(
                    f"Deleted logs older than {cutoff_date} from table {table_name}")
            except Exception as e:
                print(f"Error deleting logs from table {table_name}: {e}")
