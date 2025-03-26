from backend.disdrive_model import DisdriveModel
import json
import asyncio
import websockets


class WebsocketService:
    def __init__(self, disdrive_model: DisdriveModel):
        print("Starting websocket service...")
        # Collection of clients
        self.livefeed_clients = set()
        self.disdrive_app_clients = set()
        self.disdrive_model = disdrive_model

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
                await asyncio.sleep(0.05)
            # If client is disconnected
        except websockets.ConnectionClosed:
            print(f"Client {client_address} disconnected")
        finally:
            self.livefeed_clients.remove(client)

    async def disdrive_app_socket(self, client):
        """Socket connection for Disdrive Application; sends and retrieves settings set on application"""
        client_address = f"{client.remote_address[0]}:{client.remote_address[1]}"

        # Adding  Client
        print(f"Client {client_address} connected to Disdrive!")
        self.disdrive_app_clients.add(client)
