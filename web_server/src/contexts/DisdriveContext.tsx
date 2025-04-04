import { createContext, useState, useContext, useRef, useEffect, useCallback } from "react";

interface DisdriveContextType {
  is_logging: boolean;
  setIsLogging: (value: boolean) => void;
  has_ongoing_session: boolean;
  setHasOngoingSession: (value: boolean) => void;
  sendMessage: (value: { action: string; data?: any }) => void;
  cameras: number[];
  setCameras: (value: number[]) => void;
  camera_id: number;
  setSelectedCamera: (value: number) => void;
  session_start: string;
  setSessionStart: (value: string) => void;
}

const DisdriveContext = createContext<DisdriveContextType | undefined>(
  undefined
); // or define a type for better safety

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
  const ws = useRef<WebSocket | null>(null);

  const sendMessage = useCallback((data: { action: string; data?: any }) => {
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
  }, []);

  const handleChange = useCallback((data: Record<string, string>) => {
    switch (data.action) {
      case "toggle_logging":
        setIsLogging(prev => !prev);
        break;
      case "start_session":
        setHasOngoingSession(true);
        break;
      case "stop_session":
        setHasOngoingSession(false);
        break;
      case "update_camera":
        try {
          const cameraData = JSON.parse(data.data);
          setSelectedCamera(cameraData.camera_id);
        } catch (e) {
          console.error("Failed to parse camera data:", e);
        }
        break;
      default:
        console.warn("🚫 Invalid action:", data.action);
    }
  }, []);

  useEffect(() => {
    ws.current = new WebSocket(`ws://${window.location.hostname}:8766`);

    ws.current.onopen = () => {
      console.log("✅ Disdrive Context Connected to WebSocket server");
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log(`📡 Received message from server: `, data);
        
        if (data) {
          setIsLogging(data.is_logging);
          setHasOngoingSession(data.has_ongoing_session);
          setCameras(data.cameras || []);
          setSelectedCamera(data.camera_id);
          setSessionStart(data.session_start || "");
        }
      } catch (error) {
        console.error("⚠️ Error parsing WebSocket message:", error);
      }
    };

    ws.current.onclose = (event) => {
      console.warn("🔌 WebSocket connection closed", event.reason);
    };

    ws.current.onerror = (error) => {
      console.error("⚠️ WebSocket error:", error);
    };

    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  // Add global keyboard event listener
  useEffect(() => {
    const handleGlobalKeyPress = (event: KeyboardEvent) => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        switch (event.key.toUpperCase()) {
          case 'A':
            console.log("Start key pressed");
            sendMessage({ action: "start_session" });
            break;
          case 'B':
            console.log("Pause key pressed");
            sendMessage({ action: "stop_session" });
            break;
          case 'C':
            console.log("Shutdown key pressed");
            sendMessage({ action: "shutdown_system" });
            break;
        }
      }
    };

    window.addEventListener("keydown", handleGlobalKeyPress);
    return () => window.removeEventListener("keydown", handleGlobalKeyPress);
  }, [sendMessage]);

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
      }}
    >
      {children}
    </DisdriveContext.Provider>
  );
};

// Custom hook (optional but recommended)
export const useDisdriveContext = () => {
  const context = useContext(DisdriveContext);
  if (!context) {
    throw new Error(
      "DisdriveContext must be used within a DisdriveContextProvider"
    );
  }
  return context;
};
