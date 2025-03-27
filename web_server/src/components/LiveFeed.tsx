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
    <div className="d-flex flex-column text-center bg-white w-100">
      {imageSrc ? (
        <img src={imageSrc} alt="Live Feed" className="img-fluid mx-auto" />
      ) : (
        <p className="text-gray-500">Connecting to camera...</p>
      )}
      {GetLabels(behavior)}
    </div>
  );
};

function GetLabels(behavior: string) {
  return (
    <div className="d-flex flex-row w-100">
      <div className="flex-column text-center w-100 justify-content-center align-items-center">
        <p>
          Status:{" "}
          {behavior === "Detection Paused" ||
          behavior === "Waiting for detection..."
            ? "Detection Paused"
            : behavior === "Safe Driving"
            ? "Not Distracted"
            : "Distracted"}
        </p>
        <p>Behavior: {behavior}</p>
      </div>
      <div className="d-flex w-100 text-center bg-primary justify-content-center align-items-center">
        <p>Time Elapsed: asdasd</p>
      </div>
    </div>
  );
}

export default LiveFeed;
