import { useEffect, useState } from "react";

const useWebSocket = (url: string) => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [behavior, setBehavior] = useState<string>("Waiting for data...");

  useEffect(() => {
    const socket = new WebSocket(url);

    socket.onopen = () => console.log("Connected to WebSocket server");

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.frame) {
          setImageSrc(`data:image/jpeg;base64,${data.frame}`);
        }
        if (data.behavior) {
          setBehavior(data.behavior);
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    socket.onclose = () => console.log("WebSocket connection closed");

    return () => socket.close();
  }, [url]);

  return { imageSrc, behavior };
};

export default useWebSocket;