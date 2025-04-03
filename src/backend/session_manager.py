import asyncio
import os
import signal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - SessionManager: %(message)s'
)

class SessionManager:
    def __init__(self):
        """Manages session states and operations."""
        self.is_paused = False  # Start not paused
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Initially set (not paused) to allow camera to open
        logging.info("SessionManager initialized in running state")

    async def pause_operations(self):
        """Pauses all operations."""
        if not self.is_paused:
            logging.info("Pausing all operations...")
            self.is_paused = True
            self.pause_event.clear()
            logging.info("All operations paused successfully")
            return True
        logging.debug("Operations were already paused")
        return False

    async def resume_operations(self):
        """Resumes all operations."""
        if self.is_paused:
            logging.info("Resuming all operations...")
            self.is_paused = False
            self.pause_event.set()
            logging.info("All operations resumed successfully")
            return True
        logging.debug("Operations were already running")
        return False

    async def wait_if_paused(self):
        """Waits if operations are paused."""
        if self.is_paused:
            logging.debug("Operation waiting due to pause state")
            await self.pause_event.wait()

    def shutdown_system(self):
        """Shuts down the system/program."""
        logging.info("System shutdown initiated...")
        try:
            # Perform cleanup tasks here if needed
            logging.info("Sending termination signal...")
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            # Forceful exit as fallback
            os._exit(1)
