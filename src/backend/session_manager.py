import asyncio
import os
import signal
import logging
import pygame
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - SessionManager: %(message)s'
)

def play_sound(file_path: str, channel_id=0):
    """play sound on a specific channel to prevent cutting off other sounds"""
    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # Create up to 8 channels (different sounds)
    if channel_id >= pygame.mixer.get_num_channels():
        pygame.mixer.set_num_channels(channel_id + 1)

    # Get specific channel
    channel = pygame.mixer.Channel(channel_id)

    # Load and play sound
    sound = pygame.mixer.Sound(file_path)
    channel.play(sound)

    # Only wait for completion if specifically requested
    if channel_id == 0:
        while channel.get_busy():
            pygame.time.wait(100)  # Check less frequently to reduce CPU usage

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
        threading.Thread(target=play_sound, args=("src/assets/end.mp3", 0), daemon=True).start()
        time.sleep(5)
        try:
            # Perform cleanup tasks here if needed
            logging.info("Sending termination signal...")
            # sudo_password = os.environ.get('SUDO_PASSWORD', 'Disdrive1234')
            # try:
            #     child = pexpect.spawn('sudo shutdown -h now')
            #     child.expect('password')
            #     child.sendline(sudo_password)
            #     child.expect(pexpect.EOF)
            # except Exception as e:
            #     print(f"Error shutting down system: {e}")
            # sys.exit(0)
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            # Forceful exit as fallback
            #os._exit(1)

    def restart_system(self):
        """Restarts the system/program."""
        logging.info("System restart initiated...")
        threading.Thread(target=play_sound, args=("src/assets/restart.mp3", 0), daemon=True).start()
        time.sleep(6)
        try:
            # Perform cleanup tasks here if needed
            logging.info("Sending termination signal...")
            # sudo_password = os.environ.get('SUDO_PASSWORD', 'Disdrive1234')
            # try:
            #     child = pexpect.spawn('sudo shutdown -r now')
            #     child.expect('password')
            #     child.sendline(sudo_password)
            #     child.expect(pexpect.EOF)
            # except Exception as e:
            #     print(f"Error restarting system: {e}")
            # sys.exit(0)
        except Exception as e:
            logging.error(f"Error during restart: {e}")
            # Forceful exit as fallback
            #os._exit(1)