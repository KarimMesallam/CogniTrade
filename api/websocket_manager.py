import asyncio
import json
from typing import Dict, List, Any
from fastapi import WebSocket

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_ids: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id:
            self.connection_ids[websocket] = client_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_ids:
            del self.connection_ids[websocket]

    async def send_personal_message(self, message: Any, websocket: WebSocket):
        if isinstance(message, dict):
            await websocket.send_json(message)
        else:
            await websocket.send_text(str(message))

    async def broadcast(self, message: Any):
        for connection in self.active_connections:
            try:
                if isinstance(message, dict):
                    await connection.send_json(message)
                else:
                    await connection.send_text(str(message))
            except Exception:
                # Remove connection that might be closed or errored
                self.disconnect(connection)

    async def broadcast_to_group(self, message: Any, group_ids: List[str]):
        for connection in self.active_connections:
            try:
                if connection in self.connection_ids:
                    client_id = self.connection_ids[connection]
                    if client_id in group_ids:
                        if isinstance(message, dict):
                            await connection.send_json(message)
                        else:
                            await connection.send_text(str(message))
            except Exception:
                # Remove connection that might be closed or errored
                self.disconnect(connection)

# Create a global instance
manager = WebSocketManager() 