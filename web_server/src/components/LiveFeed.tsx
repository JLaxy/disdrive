import { useEffect, useState } from "react";

interface WebSocketData {
  frame: string;
  behavior: string;
}

const LiveFeed: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [behavior, setBehavior] = useState<string>("Waiting for detection...");

  useEffect(() => {
    const socket = new WebSocket(`ws://${window.location.hostname}:8765`);

    socket.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketData = JSON.parse(event.data);
        setImageSrc(`data:image/jpeg;base64,${data.frame}`);
        setBehavior(data.behavior);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    socket.onclose = () => console.log("WebSocket connection closed");

    return () => socket.close();
  }, []);

  return (
    <div className="p-4 text-center">
      <h2 className="text-xl font-bold mb-4">Live Driver Monitoring</h2>
      {imageSrc ? (
        <img
          src={imageSrc}
          alt="Live Feed"
          className="rounded-md shadow-md mx-auto"
        />
      ) : (
        <p className="text-gray-500">Connecting to camera...</p>
      )}
      <h3 className="mt-4 text-lg font-semibold text-red-500">{behavior}</h3>
    </div>
  );
};

export default LiveFeed;
