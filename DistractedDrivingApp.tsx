import React, { useRef, useState, useEffect } from "react";
import { Container, Button, Card, Row, Col } from "react-bootstrap";

const DistractedDrivingApp: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [behavior, setBehavior] = useState<string>("Unknown");

  // Request camera permissions on component mount
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        console.log("Camera access granted");
        stream.getTracks().forEach(track => track.stop()); // Close immediately
      })
      .catch((err) => {
        console.error("Camera access denied:", err);
        alert("⚠️ Please allow camera access in your browser settings.");
      });
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing the camera:", err);
      alert("⚠️ Unable to access the camera. Please check camera permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const handleStartSession = async () => {
    stopCamera(); // Ensure previous session is cleared
    setIsSessionActive(true);
    await startCamera();
  };

  const handleStopSession = () => {
    setIsSessionActive(false);
    stopCamera();
  };

  useEffect(() => {
    if (!isSessionActive) return;

    console.log("Connecting to WebSocket...");
    const socket = new WebSocket("ws://127.0.0.1:8765");

    socket.onopen = () => {
      console.log("WebSocket connected");
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("Received WebSocket message:", data);
        setBehavior(data.behavior || "Unknown");
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setBehavior("Error");
    };

    socket.onclose = () => {
      console.log("WebSocket closed");
    };

    return () => {
      socket.close();
    };
  }, [isSessionActive]);

  return (
    <Container className="text-center mt-5">
      <Card className="p-4">
        <h2>DISDRIVE</h2>
        <h5>DISTRACTED DRIVING DETECTION</h5>

        {!isSessionActive ? (
          <Button variant="primary" onClick={handleStartSession}>
            START SESSION
          </Button>
        ) : (
          <Row className="mt-3">
            <Col md={6} className="d-flex justify-content-center">
              <div className="video-container">
                <video ref={videoRef} autoPlay playsInline width="100%" height="300" />
              </div>
            </Col>

            <Col md={6} className="d-flex flex-column justify-content-center">
              <h5>STATUS: <span className="text-danger">Distracted</span></h5>
              <h5>BEHAVIOR: <span className="text-danger">{behavior}</span></h5>
              <h5>TIME ELAPSED: -H -M</h5>
              <Button variant="danger" className="mt-3" onClick={handleStopSession}>
                STOP SESSION
              </Button>
            </Col>
          </Row>
        )}
      </Card>
    </Container>
  );
};

export default DistractedDrivingApp;
