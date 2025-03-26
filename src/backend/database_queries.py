from backend.database_manager import DatabaseManager


class DatabaseQueries:
    """Class handling all database queries to Disdrive Database"""

    def __init__(self, path_to_db):
        print("Initializing DatabaseQueries...")
        self.db_manager: DatabaseManager = DatabaseManager(path_to_db)

    # SESSION TABLE QUERIES

    # Start a session and return the new session_id
    def start_session(self):
        self.db_manager.insert("INSERT INTO session (session_start) VALUES (CURRENT_TIMESTAMP)", ())
        return self.db_manager.fetch_one("SELECT last_insert_rowid()")  # Retrieve last inserted session_id

    # Retrieve the latest session ID
    def get_latest_session_id(self):
        return self.db_manager.fetch_one("SELECT session_id FROM session ORDER BY session_id DESC LIMIT 1")

    # Retrieve session start time for a given session_id
    def retrieve_start_session(self, session_id):
        return self.db_manager.fetch_one("SELECT session_start FROM session WHERE session_id = ?", (session_id,))

    # Update session_end for an active session
    def end_session(self, session_id):
        self.db_manager.update("UPDATE session SET session_end = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))

    # Retrieve session end time for a given session_id
    def retrieve_end_session(self, session_id):
        return self.db_manager.fetch_one("SELECT session_end FROM session WHERE session_id = ?", (session_id,))

    # Get full session details
    def get_session(self, session_id):
        return self.db_manager.fetch_one("SELECT * FROM session WHERE session_id = ?", (session_id,))


    # LOGGED BEHAVIORS QUERIES
    
    # Log a new behavior in the logged_behaviors table
    def log_behavior(self, behavior_id, session_id):
        self.db_manager.insert(
            "INSERT INTO logged_behaviors (behavior_id, session_id, behavior_time_start) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (behavior_id, session_id)
        )
        return self.db_manager.fetch_one("SELECT last_insert_rowid()")  # Get logged_behavior_id

    # Retrieve logged behaviors for a session
    def get_logged_behaviors_by_session(self, session_id):
        return self.db_manager.fetch_all("SELECT * FROM logged_behaviors WHERE session_id = ?", (session_id,))

    # Update session_time_end for a logged behavior
    def update_behavior_time_end(self, logged_behavior_id):
        self.db_manager.update(
            "UPDATE logged_behaviors SET session_time_end = CURRENT_TIMESTAMP WHERE logged_behavior_id = ?",
            (logged_behavior_id,)
        )

    # BEHAVIORS TABLE QUERIES
    
    # Get behavior_id for a specific behavior name
    def get_behavior_id(self, behavior_name):
        return self.db_manager.fetch_one("SELECT behavior_id FROM behaviors WHERE behavior = ?", (behavior_name,))

    # Get all behaviors
    def get_all_behaviors(self):
        return self.db_manager.fetch_all("SELECT * FROM behaviors")
