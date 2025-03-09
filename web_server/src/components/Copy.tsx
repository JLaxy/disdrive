import { Container, Row, Col } from "react-bootstrap";

export default function CopyComponent() {
  return (
    <Container style={{ border: "5px solid black" }}>
      <Row>
        <Col xs={12} md={6} lg={4}>
          Column 1
        </Col>
        <Col xs={12} md={6} lg={4}>
          Column 2
        </Col>
        <Col xs={12} md={12} lg={4}>
          Column 3
        </Col>
      </Row>
    </Container>
  );
}
