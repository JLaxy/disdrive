import { createContext, useState, useContext, useRef, useEffect } from "react";

interface DisdriveContextType {
  is_logging: boolean;
  setIsLogging: (value: boolean) => void;
  has_ongoing_session: boolean;
  setHasOngoingSession: (value: boolean) => void;
  sendMessage: (value: Record<string, string>) => void;
  cameras: number[];
  setCameras: (value: number[]) => void;
  camera_id: number;
  setSelectedCamera: (value: number) => void;
  session_start: string;
  setSessionStart: (value: string) => void;
  camera_view: string; // Added for camera view
  setCameraView: (value: string) => void; // Added setter for camera view
}

const DisdriveContext = createContext<DisdriveContextType | undefined>(
  undefined
);

export const DisdriveProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [is_logging, setIsLogging] = useState(true);
  const [has_ongoing_session, setHasOngoingSession] = useState(true);
  const [cameras, setCameras] = useState<number[]>([]);
  const [camera_id, setSelectedCamera] = useState<number>(0);
  const [session_start, setSessionStart] = useState<string>("");
  const [camera_view, setCameraView] = useState<string>("Front"); // Default to "Front"

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(`ws://${window.location.hostname}:8766`);

    ws.current.onopen = () => {
      console.log("✅ Disdrive Context Connected to WebSocket server");
    };

    ws.current.onmessage = (event) => {
      try {
        const data: DisdriveContextType = JSON.parse(event.data);
        console.log(`📡 Received message from server: `, data);
        setIsLogging(data.is_logging);
        setHasOngoingSession(data.has_ongoing_session);
        setCameras(data.cameras);
        setSelectedCamera(data.camera_id);
        setSessionStart(data.session_start);
        if (data.camera_view) setCameraView(data.camera_view); // Sync camera_view
      } catch (error) {
        console.error("⚠️ Error parsing WebSocket message:", error);
      }
    };

    ws.current.onclose = (event) => {
      console.warn(
        "🔌 WebSocket connection from front-end closed",
        event.reason
      );
    };

    ws.current.onerror = (error) => {
      console.error("⚠️ WebSocket error:", error);
    };

    return () => {
      ws.current?.close();
    };
  }, []);

  const sendMessage = (data: Record<string, string>) => {
    console.log(`sending ${JSON.stringify(data)} to server...`);
    try {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify(data));
        handleChange(data);
      } else {
        console.warn("🚫 WebSocket is not open. Unable to send message.");
      }
    } catch (e) {
      console.error("⚠️ Error sending message to WebSocket server:", e);
    }
  };

  const handleChange = (data: Record<string, string>) => {
    switch (data.action) {
      case "toggle_logging":
        setIsLogging(is_logging ? false : true);
        break;
      case "start_session":
        setHasOngoingSession(true);
        break;
      case "stop_session":
        setHasOngoingSession(false);
        break;
      case "update_camera":
        setSelectedCamera(JSON.parse(data.data).camera_id);
        break;
      case "update_camera_view": // Added for camera view
        setCameraView(JSON.parse(data.data).camera_view);
        break;
      default:
        console.warn("🚫 Invalid action:", data.action);
    }
  };

  return (
    <DisdriveContext.Provider
      value={{
        is_logging,
        setIsLogging,
        has_ongoing_session,
        setHasOngoingSession,
        sendMessage,
        cameras,
        setCameras,
        camera_id,
        setSelectedCamera,
        session_start,
        setSessionStart,
        camera_view, // Added for camera view
        setCameraView, // Added setter for camera view
      }}
    >
      {children}
    </DisdriveContext.Provider>
  );
};

export const useDisdriveContext = () => {
  const context = useContext(DisdriveContext);
  if (!context) {
    throw new Error(
      "DisdriveContext must be used within a DisdriveContextProvider"
    );
  }
  return context;
};
