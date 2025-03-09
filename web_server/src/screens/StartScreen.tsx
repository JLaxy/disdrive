import { Button, Container, Stack } from "react-bootstrap";
import { Link } from "react-router";

function StartScreen() {
  return (
    <Container className="bg-warning-subtle vh-100 w-100">
      <Stack gap={3}>
        <Link to={"/start_session"}>
          <Button variant="primary">Start Session</Button>
        </Link>
        <Button variant="secondary">Logs</Button>
        <Button variant="secondary">Settings</Button>
        <Button variant="outline-danger">Quit</Button>
      </Stack>
    </Container>
  );
}

export default StartScreen;
