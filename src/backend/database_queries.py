from database_manager import DatabaseManager


class DatabaseQueries:
    """Class handling all database queries to Disdrive Database"""
    def __init__(self, path_to_db):
        print("Initializing DatabaseQueries...")
        self.db_manager = DatabaseManager(path_to_db)
