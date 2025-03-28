from typing import Any
from backend.database_manager import DatabaseManager


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
                "has_ongoing_session": True
            }

        settings = {
            "is_logging": bool(result[0]),
            "camera_id": result[1],
            "has_ongoing_session": bool(result[2])
        }

        return settings

    def auto_start_session(self):
        """Auto start session on app start"""
        self.update_setting('has_ongoing_session', True)

    def log_new_session(self, date: str):
        """Logs new session on database; returns session ID"""
        print(f"logging date: {date} type: {type(date)}")

        self.db_manager.insert(
            "INSERT INTO sessions (session_start) VALUES (?)", (date,))

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
            'has_ongoing_session': 'has_ongoing_session'
        }

        # Validate the setting key
        if key not in setting_map:
            raise ValueError(f"Invalid setting key: {key}")

        # Prepare the query
        query = f"UPDATE disdrive_settings SET {setting_map[key]} = ?"

        # Execute the update
        self.db_manager.update(query, (value,))
