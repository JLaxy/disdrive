import { useEffect, useState, useRef } from "react";

interface WebSocketData {
  frame: string;
  behavior: string;
  camera: string;
}

const LiveFeed: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [behavior, setBehavior] = useState<string>("Waiting for detection...");
  const [activeCamera, setActiveCamera] = useState<string>("Front View");
  const socketRef = useRef<WebSocket | null>(null);

  // Load saved camera preference
  useEffect(() => {
    const savedSettings = localStorage.getItem("settings");
    if (savedSettings) {
      const { camera } = JSON.parse(savedSettings);
      setActiveCamera(camera || "Front View");
    }
  }, []);

  useEffect(() => {
    // Close existing socket if it exists
    if (socketRef.current) {
      socketRef.current.close();
    }

    // Create new WebSocket with selected camera
    const socket = new WebSocket(`ws://${window.location.hostname}:8765?camera=${activeCamera}`);
    socketRef.current = socket;

    socket.onopen = () => {
      // Send camera selection message
      socket.send(JSON.stringify({ camera: activeCamera }));
    };

    socket.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketData = JSON.parse(event.data);
        
        // Only update if the camera in the message matches the selected camera
        if (data.camera === activeCamera) {
          setImageSrc(`data:image/jpeg;base64,${data.frame}`);
          setBehavior(data.behavior);
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [activeCamera]);

  return (
    <div className="d-flex flex-column text-center bg-white w-100">
      <div className="p-2 bg-light">
        <h5>Active Camera: {activeCamera}</h5>
      </div>
      {imageSrc ? (
        <img src={imageSrc} alt="Live Feed" className="img-fluid mx-auto" />
      ) : (
        <p className="text-gray-500">Connecting to {activeCamera} feed...</p>
      )}
      {GetLabels(behavior)}
    </div>
  );

  function GetLabels(behavior: string) {
    return (
      <div className="d-flex flex-row w-100">
        <div className="flex-column text-center w-100 justify-content-center align-items-center">
          <p>
            Status:{" "}
            {behavior === "Detection Paused" || behavior === "Waiting for detection..."
              ? "Detection Paused"
              : behavior === "Safe Driving"
              ? "Not Distracted"
              : "Distracted"}
          </p>
          <p>Behavior: {behavior}</p>
        </div>
        <div className="d-flex w-100 text-center bg-primary justify-content-center align-items-center">
          <p>Time Elapsed: --:--</p>
        </div>
      </div>
    );
  }
};

export default LiveFeed;
