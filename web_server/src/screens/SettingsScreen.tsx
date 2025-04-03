import { Button, Card, Col, Container, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router";
import CameraDropDown from "../components/CameraDropDown";
import ViewDropDown from "../components/ViewDropDown";
import { useDisdriveContext } from "../contexts/DisdriveContext";
import { ArrowLeftIcon } from "@primer/octicons-react";

function SettingsScreen() {
  const navigate = useNavigate();
  // Checkbox state
  return (
    <Container className=" min-vh-100 d-flex align-items-center justify-content-center">
      <Card className="gap-2 p-5 w-75">
        <Row>
          <Col>
            <ArrowLeftIcon size={24} onClick={() => navigate("/")} style={{ cursor: 'pointer' }} />
          </Col>
          <Col>
            <h2 className="mb-4">Settings</h2>
          </Col>
        </Row>
        {GetCheckBox("logging", "Enable Logging")}
        <CameraDropDown />
        <ViewDropDown />
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

export default SettingsScreen;
