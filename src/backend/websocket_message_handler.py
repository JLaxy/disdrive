import asyncio
import json
from typing import Dict, Any
from backend.disdrive_model import DisdriveModel
from backend.database_queries import DatabaseQueries


class MessageHandler:
    def __init__(self, disdrive_model: DisdriveModel, database_queries: DatabaseQueries):
        """Handles incoming WebSocket messages from clients"""
        print("Initializing MessageHandler...")
        self.disdrive_model = disdrive_model
        self.database_queries = database_queries

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

            print(f"WEBSOCKETMESSAGEHANDLER action: {action}")
            print(f"WEBSOCKETMESSAGEHANDLER data: {data} type: {type(data)}")

            # checks if action is not empty
            if not action:
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
                'toggle_logging': self.toggle_logging
            }

            # Find and call the appropriate handler
            handler = action_handlers.get(action)

            # If handler is valid, call it with the data
            if handler:
                # Call handler
                response = handler(data, websocket_service)
                # Broadcast settings change to clients
                asyncio.create_task(websocket_service.broadcast_settings())
                return response
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown action: {action}'
                }

        except json.JSONDecodeError:
            return {
                'status': 'error',
                'message': 'Invalid JSON format'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Unexpected error: {str(e)}'
            }

    def update_settings(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
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

    def start_session(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """
        Start a new detection session

        Args:
            data: Optional session configuration

        Returns:
            Response with session start status
        """
        try:
            print('Starting new session MESSAGE HANDLER...')

            # Set has_ongoing_session to True
            self.database_queries.update_setting('has_ongoing_session', True)
            # Sync with model
            self.disdrive_model.update_session_status()

            return {
                'status': 'success',
                'message': 'Session started successfully'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to start session: {str(e)}'
            }

    def stop_session(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """
        Stop the current detection session

        Args:
            data: Optional session end configuration

        Returns:
            Response with session stop status
        """
        try:
            print('Stopping session MESSAGE HANDLER...')

            # Set has_ongoing_session to False
            self.database_queries.update_setting('has_ongoing_session', False)
            # Sync with model
            self.disdrive_model.update_session_status()

            return {
                'status': 'success',
                'message': 'Session stopped successfully'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to stop session: {str(e)}'
            }

    def update_camera(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
        """
        Update camera settings

        Args:
            data: Camera configuration details

        Returns:
            Response with camera update status
        """
        try:
            data = json.loads(data)

            print(f"WEBSOCKETMESSAGEHANDLER data: {data} type: {type(data)}")

            camera_id = data.get('camera_id')

            print(f"WEBSOCKETMESSAGEHANDLER camera_id: {camera_id}")

            if camera_id == None:
                print("NOT?!?!?!?!??!!??!")
                return {
                    'status': 'error',
                    'message': 'No camera ID provided'
                }

            print(f'Updating camera to: {camera_id}')

            # Update camera setting
            self.database_queries.update_setting('camera_id', camera_id)

            return {
                'status': 'success',
                'message': f'Camera updated to {camera_id}'
            }
        except Exception as e:
            print(f"ERROR!! {e}")
            return {
                'status': 'error',
                'message': f'Failed to update camera: {str(e)}'
            }

    def toggle_logging(self, data: Dict[str, Any], websocket_service) -> Dict[str, Any]:
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
            print(f'Toggling logging to: {is_logging}')

            # Update logging setting
            self.database_queries.update_setting('is_logging', is_logging)

            return {
                'status': 'success',
                'message': f'Logging {"enabled" if is_logging else "disabled"}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to toggle logging: {str(e)}'
            }
