import { useEffect, useState, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Container, Row, Col, Card, Button, Spinner, Alert } from "react-bootstrap";
import { ArrowLeftIcon } from "@primer/octicons-react";

interface Distraction {
  time: string;
  type: string;
  duration: string;
}

interface LogDetails {
  session_id: number;
  session_start: string;
  session_end: string | null;
  distractions: Distraction[];
  behaviors: Behavior[]; // Add behaviors array
}

interface Behavior {
  behavior_id: number;
  behavior_time_start: string;
  behavior_time_end: string;
  type: string;
}

function DetailedLogs() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [logDetails, setLogDetails] = useState<LogDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!id) {
      setError("Invalid session ID");
      setIsLoading(false);
      return;
    }

    const ws = new WebSocket(`ws://localhost:8000/ws/logs/${id}`);

    ws.onopen = () => {
      console.log(`Connected to WebSocket for session ${id}`);
      ws.send("get_details");
    };

    ws.onmessage = (event) => {
      try {
        const data: LogDetails = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
        } else {
          setLogDetails(data);
        }
        setIsLoading(false);
      } catch (error) {
        console.error("Failed to parse session details:", error);
        setError("Failed to parse session details");
      }
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => ws.close(); // Close WebSocket when component unmounts
  }, [id]);

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
      <Container className="text-center mt-4">
        <Spinner animation="border" variant="primary" />
        <p>Loading session details...</p>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="mt-4">
        <Alert variant="danger">{error}</Alert>
        <Button variant="secondary" onClick={() => navigate("/logs")}>
          Go Back
        </Button>
      </Container>
    );
  }

  if (!logDetails) {
    return (
      <Container className="mt-4">
        <Alert variant="danger">Session not found</Alert>
        <Button variant="secondary" onClick={() => navigate("/logs")}>
          Go Back
        </Button>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <Row className="mb-4">
        <Col xs={1}>
          <ArrowLeftIcon size={24} onClick={() => navigate("/logs")} style={{ cursor: "pointer" }} />
        </Col>
        <Col className="text-center">
          <h2>Session #{logDetails.session_id} Details</h2>
        </Col>
      </Row>

      <Card className="mb-3">
        <Card.Body>
          <Card.Title>Session Details</Card.Title>
          <Card.Text>
            <strong>Started:</strong> {formatDate(logDetails.session_start)}
            <br />
            <strong>Status:</strong> {logDetails.session_end ? (
              <span className="text-success">Completed ({formatDate(logDetails.session_end)})</span>
            ) : (
              <span className="text-primary">Ongoing</span>
            )}
          </Card.Text>
        </Card.Body>
      </Card>

      <h4 className="fw-bold mt-4">Behaviors</h4>
      {logDetails.behaviors.length > 0 ? (
        logDetails.behaviors.map((behavior, index) => (
          <Card key={index} className="mb-2">
            <Card.Body>
              <Card.Text>
                <strong>Type:</strong> {behavior.type}
                <br />
                <strong>Started:</strong> {formatDate(behavior.behavior_time_start)}
                <br />
                <strong>Ended:</strong> {formatDate(behavior.behavior_time_end)}
              </Card.Text>
            </Card.Body>
          </Card>
        ))
      ) : (
        <p className="text-muted">No behaviors recorded.</p>
      )}

      <h4 className="fw-bold">Distraction Logs</h4>
      {logDetails.distractions.length > 0 ? (
        logDetails.distractions.map((distraction, index) => (
          <Card key={index} className="mb-2">
            <Card.Body>
              <Card.Text>
                <strong>Time:</strong> {formatDate(distraction.time)}
                <br />
                <strong>Type:</strong> {distraction.type}
                <br />
                <strong>Duration:</strong> {distraction.duration}
              </Card.Text>
            </Card.Body>
          </Card>
        ))
      ) : (
        <p className="text-muted">No distractions recorded.</p>
      )}

      <Button variant="secondary" className="mt-3" onClick={() => navigate("/logs")}>
        Go Back
      </Button>
    </Container>
  );
}

export default DetailedLogs;
