import { ArrowLeftIcon } from "@primer/octicons-react";
import React from "react";
import { Card, Col, Container, Row } from "react-bootstrap";
import { useNavigate } from "react-router-dom";

// Define types
interface DistractionLog {
  time: string;
  type: string;
  duration: string;
}

interface Log {
  id: number;
  timestamp: string;
  session_end: string | null;
  distractions: DistractionLog[];
}

// Hardcoded sample data
const sampleLogs: Log[] = [
  {
    id: 1,
    timestamp: "2024-04-01T09:00:00",
    session_end: "2024-04-01T10:30:00",
    distractions: [
      { time: "5:45PM", type: "Drinking", duration: "1 Minute" },
      { time: "6:00PM", type: "Texting", duration: "2 Minutes" }
    ]
  },
  {
    id: 2,
    timestamp: "2024-04-01T11:00:00",
    session_end: "2024-04-01T12:15:00",
    distractions: []
  },
  {
    id: 3,
    timestamp: "2024-04-01T14:00:00",
    session_end: null,
    distractions: []
  }
];

function LogsScreen() {
  const navigate = useNavigate();

  // Format date for display
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <Container className="mt-4">
      <Row className="mb-4">
        <Col xs={1}>
          <ArrowLeftIcon 
            size={24} 
            onClick={() => navigate("/")} 
            style={{ cursor: 'pointer' }} 
          />
        </Col>
        <Col className="text-center">
          <h2>Session Logs</h2>
        </Col>
      </Row>

      {sampleLogs.map((log) => (
        <Card
          key={log.id}
          className="mb-3"
          onClick={() => navigate(`/logs/${log.id}`)}
          style={{
            cursor: "pointer",
            transition: "all 0.3s ease",
            borderLeft: `5px solid ${log.session_end ? '#198754' : '#0d6efd'}`,
          }}
        >
          <Card.Body>
            <Card.Title>Session #{log.id}</Card.Title>
            <Card.Text>
              <strong>Started:</strong> {formatDate(log.timestamp)}<br />
              <strong>Status:</strong> {log.session_end ? 
                <span className="text-success">
                  Completed ({formatDate(log.session_end)})
                </span> : 
                <span className="text-primary">Ongoing</span>
              }
            </Card.Text>
          </Card.Body>
        </Card>
      ))}
    </Container>
  );
}

export default LogsScreen;