from backend.disdrive_model import DisdriveModel
import json
import asyncio
import websockets


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
        async with websockets.serve(self.disdrive_app_socket, ip, port):
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

        # Add client
        self.disdrive_app_clients.add(client)

        try:

            print(f"sending settings: {self._SETTINGS}")
            # Send initial settings upon connection
            await client.send(json.dumps(self._SETTINGS))

            # Start sender and receiver listeners
            sender = asyncio.create_task(self.disdrive_app_socket_send(client))
            receiver = asyncio.create_task(
                self.disdrive_app_socket_receive(client))

            # Wait until one task finishes (disconnect or error)
            done, pending = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)

        except websockets.ConnectionClosed:
            print(f"Client {client_address} disconnected from Disdrive!")
        except Exception as e:
            print(f"An error occurred in disdrive_app_socket: {e}")
        finally:
            self.disdrive_app_clients.remove(client)
            for task in pending:
                task.cancel()

    async def disdrive_app_socket_send(self, client):
        """Sends Data to clients connected to the websocket"""
        print("disdrive_app_socket_send started")

    async def disdrive_app_socket_receive(self, client):
        """Sends Data to clients connected to the websocket"""
        print("disdrive_app_socket_receive started")
