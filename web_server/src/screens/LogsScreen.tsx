import { ArrowLeftIcon } from "@primer/octicons-react";
import React, { useEffect, useState } from "react";
import { Card, Col, Container, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router-dom";

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


// Mock log data (Replace this with API or database later)
const logs = [
  { id: 1, timestamp: "2025-03-30 10:15 AM", message: "System started successfully." },
  { id: 2, timestamp: "2025-03-30 10:30 AM", message: "User login detected." },
  { id: 3, timestamp: "2025-03-30 10:45 AM", message: "Database backup completed." },
];

function LogsScreen() {
  const navigate = useNavigate();
  // Ensure the theme is applied on initial load
  useEffect(() => {
    const storedTheme = localStorage.getItem("theme") || "light";
    document.body.dataset.bsTheme = storedTheme;
  }, []);

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
          style={{ cursor: "pointer", transition: "0.3s", borderLeft: "5px solid #0d6efd" }}
        >
          <Card.Body>
            <Card.Title>{log.timestamp}</Card.Title>
            <Card.Text>{log.message}</Card.Text>
          </Card.Body>
        </Card>
      ))}
    </Container>
  );
};

export default LogsScreen;
