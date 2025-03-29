import { Button, Card, Col, Container, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router";
import CameraDropDown from "../components/CameraDropDown";
import { useDisdriveContext } from "../contexts/DisdriveContext";

function SettingsScreen() {
  // Checkbox state
  return (
    <Container className=" min-vh-100 d-flex align-items-center justify-content-center">
      <Card className="gap-2 p-5 w-75">
        <h2 className="mb-4">Settings</h2>
        {GetCheckBox("logging", "Enable Logging")}
        <CameraDropDown />
        {GetButtons()}
      </Card>
    </Container>
  );
}

function GetCheckBox(id: string, checkBoxText: string) {
  const { is_logging, sendMessage } = useDisdriveContext();

  return (
    <Form.Check type="checkbox" id={id}>
      <Form.Check.Input
        type="checkbox"
        checked={is_logging}
        onChange={() => sendMessage({ action: "toggle_logging" })}
      />
      <Form.Check.Label className="fw-semibold">
        {checkBoxText}
      </Form.Check.Label>
    </Form.Check>
  );
}

function GetButton(buttonType: string, buttonText: string) {
  const navigate = useNavigate();
  return (
    <Button
      variant={buttonType}
      className="w-100"
      onClick={() => navigate("/")}
    >
      {buttonText}
    </Button>
  );
}

function GetButtons() {
  return (
    <Row className="g-2 pt-3">
      <Col>{GetButton("primary", "Go Back")}</Col>
    </Row>
  );
}

export default SettingsScreen;
