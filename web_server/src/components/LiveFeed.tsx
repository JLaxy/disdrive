import { useEffect, useState, useRef, memo } from "react";

interface WebSocketData {
  frame: string;
  behavior: string;
}

const LiveFeed: React.FC = memo(() => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [behavior, setBehavior] = useState<string>("Waiting for detection...");
  const socketRef = useRef<WebSocket | null>(null); // Persistent WebSocket reference

  useEffect(() => {
    console.log("📌 LiveFeed MOUNTED"); // Log when component mounts

    if (socketRef.current) return; // Prevent duplicate connections

    const ws = new WebSocket(`ws://${window.location.hostname}:8765`);
    socketRef.current = ws; // Save reference

    ws.onopen = () => {
      console.log("✅ WebSocket connected");
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketData = JSON.parse(event.data);
        setImageSrc(`data:image/jpeg;base64,${data.frame}`);
        setBehavior(data.behavior);
      } catch (error) {
        console.error("❌ Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("💥 WebSocket error:", error);
    };

    ws.onclose = (event) => {
      console.warn("🔌 WebSocket closed:", event.reason);
    };

    // Cleanup function
    return () => {
      console.log("🧹 Closing WebSocket connection...");
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []); // Empty dependency array ensures it runs **only once**

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
});

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
