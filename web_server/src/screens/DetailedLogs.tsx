import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Spinner,
  Alert,
} from "react-bootstrap";
import { ArrowLeftIcon } from "@primer/octicons-react";

interface LogDetails {
  session_id: number;
  session_start: string;
  session_end: string;
  behavior_id: number;
  behavior: string;
  behavior_time_start: string;
  behavior_time_end: string;
}

function DetailedLogs() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [logDetails, setLogDetails] = useState<LogDetails[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const port = "8767";

  useEffect(() => {
    if (!id) {
      setError("Invalid session ID");
      setIsLoading(false);
      return;
    }

    const ws = new WebSocket(
      `ws://${window.location.hostname}:${port}/ws/logs/${id}`
    );

    ws.onopen = () => {
      console.log(`Connected to WebSocket for session ${id}`);
      ws.send("get_details");
    };

    ws.onmessage = (event) => {
      try {
        const data: LogDetails[] = JSON.parse(event.data);
        if (!data) {
          setError("An error has occured!");
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
      <Row className="mb-4 align-items-center">
        <Col xs={1}>
          <div onClick={() => navigate("/logs")} style={{ cursor: "pointer" }}>
            <ArrowLeftIcon size={24} />
          </div>
        </Col>
        <Col className="text-center">
          <h2>Session #{logDetails[0].session_id} Details</h2>
        </Col>
      </Row>

      <Card className="mb-3">
        <Card.Body>
          <Card.Title>Session Details</Card.Title>
          <Card.Text>
            <strong>Started:</strong> {formatDate(logDetails[0].session_start)}
            <br />
            <strong>Status:</strong>{" "}
            {logDetails[0].session_end ? (
              <span className="text-success">
                Completed ({formatDate(logDetails[0].session_end)})
              </span>
            ) : (
              <span className="text-primary">Ongoing</span>
            )}
          </Card.Text>
        </Card.Body>
      </Card>

      <h4 className="fw-bold mt-4">Behaviors</h4>
      {logDetails?.length > 0 ? (
        logDetails.map((log, index) => (
          <Card key={index} className="mb-2">
            <Card.Body>
              <Card.Text>
                <strong>Behavior:</strong> {log.behavior}
                <br />
                <strong>Started:</strong> {formatDate(log.behavior_time_start)}
                <br />
                <strong>Ended:</strong> {formatDate(log.behavior_time_end)}
              </Card.Text>
            </Card.Body>
          </Card>
        ))
      ) : (
        <p className="text-muted">No behaviors recorded.</p>
      )}

      <Button
        variant="secondary"
        className="mt-3"
        onClick={() => navigate("/logs")}
      >
        Go Back
      </Button>
    </Container>
  );
}

export default DetailedLogs;
