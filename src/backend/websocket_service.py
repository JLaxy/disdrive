from backend.disdrive_model import DisdriveModel
import json
import asyncio
import websockets

_CONNECTION_TIMEOUT = 5  # No. of seconds to check if to disconnect clients


class WebsocketService:
    def __init__(self, disdrive_model: DisdriveModel, _SETTINGS: dict):
        print("Starting websocket service...")

        # Collection of clients
        self.livefeed_clients = set()
        self.disdrive_app_clients = set()

        # Syncing references
        self.disdrive_model: DisdriveModel = disdrive_model
        self._SETTINGS: dict = _SETTINGS

    async def start_livefeed_socket(self, ip: str, port: int):
        """Opens Live Feed socket"""
        # Start Live Feed websocket
        async with websockets.serve(self.livefeed_socket, ip, port):
            print(f"✅ Live Feed WebSocket Server started on ws://{ip}:{port}")
            await asyncio.Future()

    async def start_disdrive_app_socket(self, ip: str, port: int):
        """Opens Disdrive App socket"""
        # Start Live Feed websocket
        async with websockets.serve(self.disdrive_app_socket, ip, port, ping_interval=3, ping_timeout=5):
            print(
                f"✅ DisDrive App WebSocket Server started on ws://{ip}:{port}")

            await asyncio.Future()

    async def livefeed_socket(self, client):
        """Socket connection for the Live Feed on Session Screen; sends data to client front-end"""
        client_address = f"{client.remote_address[0]}:{client.remote_address[1]}"

        # Adding Client
        print(f"Client {client_address} connected to Live Feed!")
        self.livefeed_clients.add(client)

        try:
            # Loop while client is still connected
            while True:
                livefeed_data = json.dumps(
                    self.disdrive_model.latest_detection_data)

                # Send data to frontend clients
                await client.send(livefeed_data)
                await asyncio.sleep(0.01)
            # If client is disconnected
        except websockets.ConnectionClosed:
            print(f"Client {client_address} disconnected from Live Feed!")
        except Exception as e:
            print(f"An error has occured in livefeed_socket: {e}")
        finally:
            self.livefeed_clients.remove(client)

    async def disdrive_app_socket(self, client):
        """Socket connection for Disdrive Application; sends and retrieves settings set on application"""
        client_address = f"{client.remote_address[0]}:{client.remote_address[1]}"
        print(f"Client {client_address} connected to Disdrive Frontend!")

        self.disdrive_app_clients.add(client)

        try:
            print(f"sending settings: {self._SETTINGS}")
            await client.send(json.dumps(self._SETTINGS))

            keepalive = asyncio.create_task(
                self.disdrive_app_socket_keepalive(client))
            receiver = asyncio.create_task(
                self.disdrive_app_socket_receive(client))

            done, pending = await asyncio.wait([keepalive, receiver], return_when=asyncio.FIRST_COMPLETED)

        except websockets.ConnectionClosed:
            print(f"Client {client_address} disconnected from Disdrive!")
        except Exception as e:
            print(f"An error occurred in disdrive_app_socket: {e}")
        finally:
            print(
                f"Client {client_address} disconnected from Disdrive!\nCleaning up...")
            self.disdrive_app_clients.discard(client)

            for task in pending:
                task.cancel()
                try:
                    await task  # Ensure proper cleanup
                except asyncio.CancelledError:
                    pass

            print(f"✅ Tasks cleaned up for {client_address}")

    async def disdrive_app_socket_keepalive(self, client):
        """Keeps WebSocket connection alive and detects disconnection."""
        print("📤 disdrive_app_socket_keepalive started")

        try:
            while True:  # Run indefinitely
                await asyncio.Future()

        except websockets.ConnectionClosedOK:
            print("✅ keepalive: Client disconnected normally.")
        except websockets.ConnectionClosed as e:
            print(
                f"❌ keepalive: Client disconnected unexpectedly (code={e.code}, reason={e.reason})")
        except Exception as e:
            print(f"⚠️ keepalive error: {e}")

    async def disdrive_app_socket_receive(self, client):
        """Receives Data from clients and detects disconnection"""
        print("📩 disdrive_app_socket_receive started")

        try:
            async for message in client:
                print(f"📨 Received: {message}")

        except websockets.ConnectionClosed as e:
            print(f"❌ Client disconnected (code={e.code}, reason={e.reason})")
            return  # 🔥 Avoid raising an unnecessary exception

        except Exception as e:
            print(f"⚠️ receive_loop error: {e}")
