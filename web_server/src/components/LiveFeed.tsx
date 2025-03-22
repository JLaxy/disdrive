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

    socket.onopen = () => {
      console.log("✅ WebSocket connection established");
    };

    socket.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketData = JSON.parse(event.data);
        setImageSrc(`data:image/jpeg;base64,${data.frame}`);
        setBehavior(data.behavior);
      } catch (error) {
        console.error("❌ Error parsing WebSocket message:", error);
      }
    };

    socket.onerror = (error) => {
      console.error("💥 WebSocket error:", error);
    };

    socket.onclose = (event) => {
      console.warn("🔌 WebSocket connection closed", event.reason);
    };

    // Cleanup on unmount
    return () => {
      console.log("🧹 Cleaning up WebSocket");
      socket.close();
    };
  }, []);

  return (
    <div className="d-flex flex-column p-4 text-center bg-warning w-100">
      <h2 className="text-xl font-bold mb-4">Live Driver Monitoring</h2>
      {imageSrc ? (
        <img src={imageSrc} alt="Live Feed" className="img-fluid mx-auto" />
      ) : (
        <p className="text-gray-500">Connecting to camera...</p>
      )}
      <h3 className="mt-4 text-lg font-semibold text-red-500">{behavior}</h3>
    </div>
  );
};

export default LiveFeed;
