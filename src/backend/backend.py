import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import sqlite3
import json
from pathlib import Path
import os
from datetime import datetime
from behaviors import Behaviors

# Initialize FastAPI app
app = FastAPI()

# Define behavior labels
_BEHAVIOR_LABEL = {
    0: "Safe Driving",
    1: "Texting",
    2: "Talking using Phone",
    3: "Drinking",
    4: "Head Down",
    5: "Look Behind"
}

# Get the absolute path to the database file
current_dir = Path(__file__).parent
DB_PATH = (current_dir.parent.parent / "database" / "disdrive_db.db").resolve()

# Create database directory if it doesn't exist
os.makedirs(DB_PATH.parent, exist_ok=True)

# Class to manage WebSocket connections


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

# Initialize database with tables if they don't exist


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start TEXT NOT NULL,
                session_end TEXT
            );

            CREATE TABLE IF NOT EXISTS distractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                time TEXT NOT NULL,
                type TEXT NOT NULL,
                duration TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


# Initialize the database
init_db()


@app.websocket("/ws/logs")
async def get_all_logs(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM sessions ORDER BY session_start DESC")
            rows = cursor.fetchall()
            logs = [
                {
                    'session_id': row[0],
                    'session_start': row[1],
                    'session_end': row[2]
                }
                for row in rows
            ]
            await websocket.send_text(json.dumps(logs))
            # Keep connection alive
            while True:
                await websocket.receive_text()
        finally:
            conn.close()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected")


@app.websocket("/ws/logs/{session_id}")
async def get_session_details(websocket: WebSocket, session_id: int):
    await manager.connect(websocket)
    print(f"📡 New connection established for session {session_id}")

    conn = None
    try:
        while True:
            message = await websocket.receive_text()
            if message == "get_details":
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                # Get session details
                cursor.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                session = cursor.fetchone()

                if session:
                    # Get distractions
                    cursor.execute("""
                        SELECT time, type, duration
                        FROM distractions
                        WHERE session_id = ?
                        ORDER BY time DESC
                    """, (session_id,))
                    distractions = [
                        {"time": d[0], "type": d[1], "duration": d[2]}
                        for d in cursor.fetchall()
                    ]

                    # Get logged behaviors
                    cursor.execute("""
                        SELECT behavior_id, behavior_time_start, behavior_time_end
                        FROM logged_behaviors
                        WHERE session_id = ?
                        ORDER BY behavior_time_start DESC
                    """, (session_id,))
                    behaviors = [
                        {
                            "behavior_id": b[0],
                            "behavior_time_start": b[1],
                            "behavior_time_end": b[2],
                            "type": _BEHAVIOR_LABEL[b[0]]
                        }
                        for b in cursor.fetchall()
                    ]

                    details = {
                        "session_id": session[0],
                        "session_start": session[1],
                        "session_end": session[2],
                        "distractions": distractions,
                        "behaviors": behaviors
                    }
                    await websocket.send_text(json.dumps(details))
    except WebSocketDisconnect:
        print(f"❌ Client disconnected from session {session_id}")
    except Exception as e:
        print(f"❌ Error in WebSocket connection: {e}")
    finally:
        if conn:
            conn.close()
        manager.disconnect(websocket)

# REST API endpoint for logs


@app.get("/api/logs")
async def get_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM sessions ORDER BY session_start DESC")
        rows = cursor.fetchall()
        return [
            {
                'session_id': row[0],
                'session_start': row[1],
                'session_end': row[2]
            }
            for row in rows
        ]
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
