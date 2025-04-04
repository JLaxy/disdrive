import asyncio
import json
import logging
from typing import Dict, Any
from backend.disdrive_model import DisdriveModel
from backend.database_queries import DatabaseQueries
from backend.session_manager import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageHandler:
    def __init__(self, disdrive_model: DisdriveModel, database_queries: DatabaseQueries, session_manager: SessionManager):
        """Handles incoming WebSocket messages from clients"""
        logger.info("Initializing MessageHandler...")
        self.disdrive_model = disdrive_model
        self.database_queries = database_queries
        self.session_manager = session_manager  # Use shared session manager

    async def process_message(self, message: str, websocket_service) -> Dict[str, Any]:
        """
        Process incoming WebSocket messages

        Args:
            message: JSON string containing message details

        Returns:
            Response dictionary with status and optional data
        """
        try:
            # Parse the incoming message
            msg_data = json.loads(message)

            # Extract action and data
            action = msg_data.get('action')
            data = msg_data.get('data', {})

            logger.info(f"Processing action: {action}")

            # Ensure data is a dictionary
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    print(f"Invalid data format: {data}")
                    return {
                        'status': 'error',
                        'message': 'Invalid data format'
                    }

            # checks if action is not empty
            if not action:
                print(f"No action specified in message: {message}")
                return {
                    'status': 'error',
                    'message': 'No action specified'
                }

            # Handle different actions using a dictionary of methods
            action_handlers = {
                'update_settings': self.update_settings,
                'start_session': self.start_session,
                'stop_session': self.stop_session,
                'update_camera': self.update_camera,
                'toggle_logging': self.toggle_logging,
                'shutdown_system': self.shutdown_system,
                'update_retention_days': self.update_retention_days
            }

            # Find and call the appropriate handler
            handler = action_handlers.get(action)

            # If handler is valid, call it with the data
            if handler:
                response = await handler(data, websocket_service)
                asyncio.create_task(websocket_service.broadcast_settings())
                logger.info(f"Handler response: {response}")
                return response
            else:
                print("error!!!")
                return {
                    'status': 'error',
                    'message': f'Unknown action: {action}'
                }

        except json.JSONDecodeError:
            print("failed to decode json!")
            return {
                'status': 'error',
                'message': 'Invalid JSON format'
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'status': 'error',
                'message': f'Unexpected error: {str(e)}'
            }
        
    async def update_retention_days(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        try:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return {'status': 'error', 'message': 'retention days data format'}

            days = data.get('retention_days')
            if days is None:
                return {'status': 'error', 'message': 'Invalid retention days'}
            
            # Update logging setting
            self.database_queries.update_setting("retention_days", days)

            return {
                'status': 'success',
                'message': f'Updated retention days to {days}'
            }
        except Exception as e:
            logger.error(f"Failed to update retention daysd")
            return {
                'status': 'error',
                'message': f'Failed to update retention days: {e}'
            }

    async def update_settings(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """
        Update multiple settings at once

        Args:
            data: Dictionary of settings to update

        Returns:
            Response with update status
        """
        try:
            # Update each setting
            for key, value in data.items():
                self.database_queries.update_setting(key, value)

            # Fetch and return updated settings
            updated_settings = self.database_queries.get_settings()

            return {
                'status': 'success',
                'message': 'Settings updated successfully',
                'settings': updated_settings
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to update settings: {str(e)}'
            }

    async def start_session(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """Start a new detection session"""
        try:
            logger.info("Starting session...")
            if await self.session_manager.resume_operations():
                self.database_queries.update_setting('has_ongoing_session', True)
                self.disdrive_model.log_manager.start_session()
                self.disdrive_model.update_session_status()
                return {
                    'status': 'success',
                    'message': 'Session started successfully'
                }
            return {
                'status': 'warning',
                'message': 'Session was already running'
            }
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return {
                'status': 'error',
                'message': f'Failed to start session: {str(e)}'
            }

    async def stop_session(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """Stop the current detection session"""
        try:
            logger.info("Stopping session...")
            if await self.session_manager.pause_operations():
                self.database_queries.update_setting('has_ongoing_session', False)
                self.disdrive_model.update_session_status()
                return {
                    'status': 'success',
                    'message': 'Session stopped successfully'
                }
            return {
                'status': 'warning',
                'message': 'Session was already stopped'
            }
        except Exception as e:
            logger.error(f"Failed to stop session: {e}")
            return {
                'status': 'error',
                'message': f'Failed to stop session: {str(e)}'
            }

    async def update_camera(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        try:
            logger.info("Updating camera...")
            
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return {'status': 'error', 'message': 'Invalid camera data format'}

            camera_id = data.get('camera_id')
            if camera_id is None:
                return {'status': 'error', 'message': 'No camera ID provided'}

            camera_id = int(camera_id)
            logger.info(f'Updating camera to: {camera_id}')

            # Change camera
            await self.disdrive_model.change_camera(camera_id)
            self.database_queries.update_setting('camera_id', camera_id)

            return {
                'status': 'success',
                'message': f'Camera updated to {camera_id}',
                'data': {'camera_id': camera_id}
            }
        except Exception as e:
            logger.error(f"Failed to update camera: {e}")
            return {'status': 'error', 'message': str(e)}

    async def toggle_logging(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """
        Toggle logging on/off

        Args:
            data: Logging configuration

        Returns:
            Response with logging toggle status
        """
        try:
            # Get opposite of current settings
            is_logging = not bool(
                websocket_service.get_updated_settings()["is_logging"])
            logger.info(f'Toggling logging to: {is_logging}')

            # Update logging setting
            self.database_queries.update_setting('is_logging', is_logging)

            return {
                'status': 'success',
                'message': f'Logging {"enabled" if is_logging else "disabled"}'
            }
        except Exception as e:
            logger.error(f"Failed to toggle logging: {e}")
            return {
                'status': 'error',
                'message': f'Failed to toggle logging: {str(e)}'
            }

    async def shutdown_system(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """Shutdown the system"""
        try:
            logger.info('Initiating system shutdown...')
            # First clean up WebSocket connections
            await websocket_service.cleanup()
            
            # Then initiate shutdown
            self.session_manager.shutdown_system()
            
            return {
                'status': 'success',
                'message': 'System shutdown initiated'
            }
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return {
                'status': 'error',
                'message': f'Failed to shutdown system: {str(e)}'
            }

