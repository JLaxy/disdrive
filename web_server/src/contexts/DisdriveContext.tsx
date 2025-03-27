import { createContext, useState, useContext, useRef, useEffect } from "react";

interface DisdriveContextType {
  is_logging: boolean;
  setIsLogging: (value: boolean) => void;
  has_ongoing_session: boolean;
  setHasOngoingSession: (value: boolean) => void;
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

  const ws = useRef<WebSocket | null>(null); // Store WebSocket instance

  useEffect(() => {
    ws.current = new WebSocket(`ws://${window.location.hostname}:8766`);

    ws.current.onopen = () => {
      console.log("✅ Disdrive Context Connected to WebSocket server");
    };

    ws.current.onmessage = (event) => {
      try {
        console.log("received message");
        const data: DisdriveContextType = JSON.parse(event.data);
        setIsLogging(data.is_logging);
        setHasOngoingSession(data.has_ongoing_session);
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

  return (
    <DisdriveContext.Provider
      value={{
        is_logging,
        setIsLogging,
        has_ongoing_session,
        setHasOngoingSession,
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
