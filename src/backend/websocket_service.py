import json
import asyncio
import websockets

_CONNECTION_TIMEOUT = 5  # No. of seconds to check if to disconnect clients

class WebsocketService:
    def __init__(self, disdrive_model, _SETTINGS):
        print("Starting websocket service...")

        # Collection of clients
        self.livefeed_clients = set()
        self.disdrive_app_clients = set()

        # Syncing references
        self.disdrive_model = disdrive_model
        self._SETTINGS = _SETTINGS

        # Flags to control server shutdown
        self.livefeed_server = None
        self.disdrive_app_server = None

    async def start_livefeed_socket(self, ip: str, port: int):
        """Opens Live Feed socket"""
        try:
            self.livefeed_server = await websockets.serve(self.livefeed_socket, ip, port)
            print(f"✅ Live Feed WebSocket Server started on ws://{ip}:{port}")
            await self.livefeed_server.wait_closed()
        except Exception as e:
            print(f"❌ Live Feed WebSocket Server error: {e}")
        finally:
            if self.livefeed_server:
                self.livefeed_server.close()

    async def start_disdrive_app_socket(self, ip: str, port: int):
        """Opens Disdrive App socket"""
        try:
            self.disdrive_app_server = await websockets.serve(self.disdrive_app_socket, ip, port, ping_interval=3, ping_timeout=5)
            print(f"✅ DisDrive App WebSocket Server started on ws://{ip}:{port}")
            await self.disdrive_app_server.wait_closed()
        except Exception as e:
            print(f"❌ DisDrive App WebSocket Server error: {e}")
        finally:
            if self.disdrive_app_server:
                self.disdrive_app_server.close()

    async def livefeed_socket(self, client):
        """Socket connection for the Live Feed on Session Screen"""
        client_address = f"{client.remote_address[0]}:{client.remote_address[1]}"

        try:
            # Adding Client
            print(f"Client {client_address} connected to Live Feed!")
            self.livefeed_clients.add(client)

            # Loop while client is still connected
            while True:
                livefeed_data = json.dumps(
                    self.disdrive_model.latest_detection_data)

                # Send data to frontend clients
                await client.send(livefeed_data)
                await asyncio.sleep(0.01)

        except websockets.ConnectionClosed:
            print(f"Client {client_address} disconnected from Live Feed!")
        except Exception as e:
            print(f"An error has occurred in livefeed_socket: {e}")
        finally:
            self.livefeed_clients.discard(client)

    async def disdrive_app_socket(self, client):
        """Socket connection for Disdrive Application"""
        client_address = f"{client.remote_address[0]}:{client.remote_address[1]}"
        print(f"Client {client_address} connected to Disdrive Frontend!")
        
        try:
            self.disdrive_app_clients.add(client)
            
            # Send settings to client
            print(f"Sending settings: {self._SETTINGS} to client {client_address}")
            await client.send(json.dumps(self._SETTINGS))

            # Handle receiving messages
            async for message in client:
                print(f"📨 Received: {message}")

        except websockets.ConnectionClosed:
            print(f"Client {client_address} disconnected from Disdrive!")
        except Exception as e:
            print(f"An error occurred in disdrive_app_socket: {e}")
        finally:
            print(f"Client {client_address} disconnected. Cleaning up...")
            self.disdrive_app_clients.discard(client)

    def stop_servers(self):
        """Gracefully stop WebSocket servers"""
        if self.livefeed_server:
            self.livefeed_server.close()
        if self.disdrive_app_server:
            self.disdrive_app_server.close()