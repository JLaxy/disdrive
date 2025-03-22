import { Button, Card, Col, Container, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router";
import CameraDropDown from "../components/CameraDropDown";
import { useState } from "react";

function SettingsScreen() {
  // Checkbox state
  const [toDistractAlarm, setDistractAlarm] = useState(false);
  const [toLogging, setLogging] = useState(false);
  return (
    <Container className="bg-warning min-vh-100 d-flex align-items-center justify-content-center">
      <Card className="gap-2 p-5 w-75">
        <h2 className="mb-4">Settings</h2>
        {GetCheckBox("dist-alarm", "Distraction Alarm", toDistractAlarm, (e) =>
          setDistractAlarm(e.target.checked)
        )}
        {GetCheckBox("logging", "Enable Logging", toLogging, (e) =>
          setLogging(e.target.checked)
        )}
        <CameraDropDown />
        {GetButtons()}
      </Card>
    </Container>
  );
}

function GetCheckBox(
  id: string,
  checkBoxText: string,
  checked: boolean,
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
) {
  return (
    <Form.Check type="checkbox" id={id}>
      <Form.Check.Input type="checkbox" checked={checked} onChange={onChange} />
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
      <Col>{GetButton("danger", "Go Back")}</Col>
      <Col>{GetButton("success", "Save")}</Col>
    </Row>
  );
}

export default SettingsScreen;
