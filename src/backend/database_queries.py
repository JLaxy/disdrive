from backend.database_manager import DatabaseManager


class DatabaseQueries:
    """Class handling all database queries to Disdrive Database"""

    def __init__(self, path_to_db):
        print("Initializing DatabaseQueries...")
        self.db_manager: DatabaseManager = DatabaseManager(path_to_db)

    def get_settings(self):
        """Returns saved settings in dictionary form from database"""
        result = self.db_manager.fetch_one(
            "SELECT * FROM disdrive_settings")

        settings = {
            "is_logging": bool(result[0]),
            "camera_id": result[1],
            "has_ongoing_session": bool(result[2])
        }

        return settings
