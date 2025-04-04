import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, Col, Container, Row, Spinner } from "react-bootstrap";
import { ArrowLeftIcon } from "@primer/octicons-react";

interface Log {
  session_id: number;
  session_start: string;
  session_end: string | null;
}

function LogsScreen() {
  const port = "8767";
  const [logs, setLogs] = useState<Log[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const ws = new WebSocket(
      `ws://${window.location.hostname}:${port}/ws/logs`
    );

    ws.onopen = () => {
      console.log("Connected to logs WebSocket");
      setIsLoading(false);
    };

    ws.onmessage = (event) => {
      try {
        const data: Log[] = JSON.parse(event.data);
        setLogs(data);
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => ws.close(); // Close WebSocket when component unmounts
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  if (isLoading) {
    return (
      <Container className="d-flex justify-content-center align-items-center text-center mt-4 min-vh-100">
        <Spinner animation="border" variant="primary" />
        <p>Loading logs...</p>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <Row className="mb-4">
        <Col xs={1}>
          <div
            onClick={() => navigate("/")}
            style={{
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
            }}
          >
            <ArrowLeftIcon size={24} />
          </div>
        </Col>
        <Col className="text-center">
          <h2>Session Logs</h2>
        </Col>
      </Row>

      {logs.map((log) => (
        <Card
          key={log.session_id}
          className="mb-3"
          onClick={() => navigate(`/logs/${log.session_id}`)} // Redirects to DetailedLogs with session ID
          style={{
            cursor: "pointer",
            transition: "all 0.3s ease",
            borderLeft: `5px solid ${log.session_end ? "#198754" : "#0d6efd"}`,
          }}
        >
          <Card.Body>
            <Card.Title>Session #{log.session_id}</Card.Title>
            <Card.Text>
              <strong>Started:</strong> {formatDate(log.session_start)}
              <br />
              <strong>Status:</strong>{" "}
              {log.session_end ? (
                <span className="text-success">
                  Completed ({formatDate(log.session_end)})
                </span>
              ) : (
                <span className="text-primary">Ongoing</span>
              )}
            </Card.Text>
          </Card.Body>
        </Card>
      ))}
    </Container>
  );
}

export default LogsScreen;
