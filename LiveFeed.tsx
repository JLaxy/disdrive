import React, { useState, useEffect } from "react";

const LiveFeed: React.FC = () => {
    const [frame, setFrame] = useState<string | null>(null);
    const [behavior, setBehavior] = useState<string>("Detecting...");

    useEffect(() => {
        const ws = new WebSocket("ws://localhost:8765");

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setFrame(`data:image/jpeg;base64,${data.frame}`);
            setBehavior(data.behavior);
        };

        ws.onerror = (error) => console.error("WebSocket Error: ", error);

        return () => ws.close(); // Cleanup on unmount
    }, []);

    return (
        <div>
            <h2>Live Feed</h2>
            {frame && <img src={frame} alt="Live Video Feed" width="640" />}
            <p><strong>Behavior:</strong> {behavior}</p>
        </div>
    );
};

export default LiveFeed;