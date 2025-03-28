import { useEffect, useState, useRef, memo } from "react";
import { useLocation } from "react-router-dom";
import { getWebSocket, closeWebSocket } from "../utils/LiveFeedSocketService";
import { useDisdriveContext } from "../contexts/DisdriveContext";
import { differenceInSeconds } from "date-fns";

interface WebSocketData {
  frame: string;
  behavior: string;
}

const LiveFeed: React.FC = memo(() => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [behavior, setBehavior] = useState<string>("Waiting for detection...");
  const wsRef = useRef<WebSocket | null>(null);
  const location = useLocation();
  const { session_start, has_ongoing_session } = useDisdriveContext();

  useEffect(() => {
    console.log("📌 LiveFeed MOUNTED");

    // Get or create WebSocket instance
    wsRef.current = getWebSocket(`ws://${window.location.hostname}:8765`);

    wsRef.current.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketData = JSON.parse(event.data);
        setImageSrc(`data:image/jpeg;base64,${data.frame}`);
        setBehavior(data.behavior);
      } catch (error) {
        console.error("❌ Error parsing WebSocket message:", error);
      }
    };

    return () => {
      console.log("🧹 LiveFeed UNMOUNTED → Checking if WebSocket should close");

      // Close WebSocket only if navigating AWAY from `/session`
      setTimeout(() => {
        if (location.pathname !== "/session") {
          console.log("🛑 Closing WebSocket because user left `/session`");
          closeWebSocket();
        }
      }, 100);
    };
  }, [location.pathname]); // Runs when route changes

  return (
    <div className="d-flex flex-column text-center bg-white w-100">
      {imageSrc ? (
        <img src={imageSrc} alt="Live Feed" className="img-fluid mx-auto" />
      ) : (
        <p className="text-gray-500">Connecting to camera...</p>
      )}
      {GetLabels(behavior, session_start, has_ongoing_session)}
    </div>
  );
});

function GetLabels(
  behavior: string,
  session_start: string,
  has_ongoing_session: boolean
) {
  const [elapsedTime, setElapsedTime] = useState("00:00:00");

  useEffect(() => {
    if (!has_ongoing_session || !session_start) return;

    const updateElapsedTime = () => {
      const startDate = new Date(session_start);
      const now = new Date();
      const totalSeconds = differenceInSeconds(now, startDate);

      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = totalSeconds % 60;

      setElapsedTime(
        `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
          2,
          "0"
        )}:${String(seconds).padStart(2, "0")}`
      );
    };

    // Update immediately and then every second
    updateElapsedTime();
    const interval = setInterval(updateElapsedTime, 1000);

    return () => clearInterval(interval);
  }, [session_start, has_ongoing_session]);

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
        <p>
          Time Elapsed: {has_ongoing_session ? elapsedTime : "Detection Paused"}
        </p>
      </div>
    </div>
  );
}

export default LiveFeed;
