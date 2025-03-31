let wsInstance: WebSocket | null = null;

export const getWebSocket = (url: string) => {
  if (!wsInstance || wsInstance.readyState === WebSocket.CLOSED) {
    console.log("⚡ Creating new WebSocket connection...");
    wsInstance = new WebSocket(url);

    wsInstance.onopen = () => console.log("✅ WebSocket connected");
    wsInstance.onclose = (event) => {
      console.warn("🔌 WebSocket closed:", event.reason);
      wsInstance = null; // Allow reconnection on next mount
    };
    wsInstance.onerror = (error) => console.error("💥 WebSocket error:", error);
  } else {
    console.log("🔄 Reusing existing WebSocket connection...");
  }
  return wsInstance;
};

export const closeWebSocket = () => {
  if (wsInstance) {
    console.log("🛑 Closing WebSocket...");
    wsInstance.close();
    wsInstance = null;
  }
};
