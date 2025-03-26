import sqlite3


class DatabaseManager:
    """Handles all database operations for the Disdrive Database"""

    def __init__(self, path_to_db: str):
        """Initialize the database path"""
        self.db_path = path_to_db

        # If unable to connect to database, exit
        if (not self.test_connect_to_db()):
            exit(1)

    def test_connect_to_db(self):
        """Tests database connection"""
        print("Testing database connection...")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            print(f"Successfully connected to database {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return False

    def connect_to_db(self):
        """Opens a new database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            return conn, cursor
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None, None

    def insert(self, query: str, params: tuple):
        """Inserts data into the database"""
        conn, cursor = self.connect_to_db()
        if conn and cursor:
            try:
                cursor.execute(query, params)
                conn.commit()
                print("Data inserted successfully.")
            except sqlite3.Error as e:
                print(f"Error inserting data: {e}")
            finally:
                conn.close()

    def update(self, query: str, params: tuple):
        """Updates data in the database"""
        conn, cursor = self.connect_to_db()
        if conn and cursor:
            try:
                cursor.execute(query, params)
                conn.commit()
                print("Data updated successfully.")
            except sqlite3.Error as e:
                print(f"Error updating data: {e}")
            finally:
                conn.close()

    def delete(self, query: str, params: tuple):
        """Deletes data from the database"""
        conn, cursor = self.connect_to_db()
        if conn and cursor:
            try:
                cursor.execute(query, params)
                conn.commit()
                print("Data deleted successfully.")
            except sqlite3.Error as e:
                print(f"Error deleting data: {e}")
            finally:
                conn.close()

    def fetch_all(self, query: str, params: tuple = ()):
        """Fetches all results from a query"""
        conn, cursor = self.connect_to_db()
        if conn and cursor:
            try:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return results
            except sqlite3.Error as e:
                print(f"Error fetching data: {e}")
                return []
            finally:
                conn.close()

    def fetch_one(self, query: str, params: tuple = ()):
        """Fetches a single row from a query"""
        conn, cursor = self.connect_to_db()
        if conn and cursor:
            try:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result
            except sqlite3.Error as e:
                print(f"Error fetching data: {e}")
                return None
            finally:
                conn.close()
