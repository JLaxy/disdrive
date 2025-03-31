import { ArrowLeftIcon } from "@primer/octicons-react";
import React, { useEffect, useState } from "react";
import { Card, Col, Container, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router-dom";

// Define the Log type
interface Log {
  id: number;
  timestamp: string;
  message: string;
  type: string;
}

const DarkModeToggle = () => {
  // State to track the dark mode status
  const storedTheme = localStorage.getItem("theme");
  const initialTheme = storedTheme ? storedTheme : "light";

  const [isDarkMode, setIsDarkMode] = useState(initialTheme === "dark");

  // Effect to apply dark mode to the body when the state changes
  useEffect(() => {
    document.body.dataset.bsTheme = isDarkMode ? "dark" : "light";
    localStorage.setItem("theme", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  return (
    <Container className="p-3 d-flex justify-content-end">
      <Row>
        <Col xs="auto">
          <div>
            <Form.Check
              type="switch"
              id="custom-switch"
              label="Dark Mode"
              checked={isDarkMode}
              onChange={() => setIsDarkMode(!isDarkMode)} // Toggle dark mode
            />
          </div>
        </Col>
      </Row>
    </Container>
  );
};

function DetailedLogs() {
  const navigate = useNavigate();
  const [logs, setLogs] = useState<Log[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch logs from the WebSocket server
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8766');

    ws.onopen = () => {
      console.log('Connected to WebSocket server');
      // Request logs data
      ws.send(JSON.stringify({ type: 'get_logs' }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'logs_data') {
          setLogs(data.logs);
          setLoading(false);
        }
      } catch (err) {
        setError('Failed to parse logs data');
        setLoading(false);
      }
    };

    ws.onerror = () => {
      setError('Failed to connect to server');
      setLoading(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  // Ensure the theme is applied on initial load
  useEffect(() => {
    const storedTheme = localStorage.getItem("theme") || "light";
    document.body.dataset.bsTheme = storedTheme;
  }, []);

  // Function to get border color based on log type
  const getLogBorderColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'error':
        return '#dc3545'; // Red
      case 'warning':
        return '#ffc107'; // Yellow
      case 'success':
        return '#198754'; // Green
      default:
        return '#0d6efd'; // Blue
    }
  };

  if (loading) {
    return (
      <Container className="mt-4">
        <div>Loading logs...</div>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="mt-4">
        <div className="text-danger">Error: {error}</div>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <Row>
        <Col>
          <ArrowLeftIcon size={24} onClick={() => navigate("/")} style={{ cursor: 'pointer' }} />
        </Col>
        <Col>
          <h2>Logs</h2>
        </Col>
        <Col>
          <DarkModeToggle />
        </Col>
      </Row>
      {logs.map((log) => (
        <Card
          key={log.id}
          className="mb-3"
          onClick={() => navigate(`/logs/${log.id}`)}
          style={{ 
            cursor: "pointer", 
            transition: "0.3s", 
            borderLeft: `5px solid ${getLogBorderColor(log.type)}`
          }}
        >
          <Card.Body>
            <Card.Title>{new Date(log.timestamp).toLocaleString()}</Card.Title>
            <Card.Text>{log.message}</Card.Text>
          </Card.Body>
        </Card>
      ))}
    </Container>
  );
};

export default DetailedLogs;
