PULL DATABASE

--------------------------------------------

FILES TO PULL (START/STOP/SHUTDOWN):

- SessionsScreen.tsx

- DisDriveContext.tsx

- websocket_message_handler.py

- session_manager.py

- LandingScreen.tsx


FUNCTIONS USED (START/STOP/SHUTDOWN)::

- start_session ()

- resume_operations ()

- stop_session ()

- pause_operations ()

- shutdown_system ()


---------------------------------------------------------

FILES TO PULL (DYNAMIC):

- log_manager.py

- database_queries.py

- disdrive.py

- DisDriveContext.tsx

- SettingsScreen.tsx


FUNCTIONS USED (DYNAMIC):

- self.delete_old_logs()  declaration in class LogManager

- def delete_old_logs(self): function for deleting logs (set temporarily to 15 days after log was added. To test, change date time of device. NOT DYNAMIC YET)

- def delete_logs_from_multiple_tables(self, table_names: list[str], cutoff_date: str): function from the database_queries.py for the query basis used in def delete_old_logs(self)

- log_manager = LogManager(database_query) just a declaration of function in disdrive.py

- added retention days function
