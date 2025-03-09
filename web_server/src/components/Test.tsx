import { Col, Container, Row } from "react-bootstrap";

export default function TestComponent() {
  return (
    <Container fluid="true" style={{ border: "5px solid black" }}>
      <Row xs={1} lg={3}>
        <Col>my 1</Col>
        <Col>my 2</Col>
        <Col>my 3</Col>
      </Row>
    </Container>
  );
}
