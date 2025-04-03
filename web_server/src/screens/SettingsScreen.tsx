import { Card, Col, Container, Form, Row, Button } from "react-bootstrap";
import { useNavigate } from "react-router";
import CameraDropDown from "../components/CameraDropDown";
import { useDisdriveContext } from "../contexts/DisdriveContext";
import { ArrowLeftIcon } from "@primer/octicons-react";
import NumberSpinner from "../components/NumberSpinner";
import { useState } from "react";

function SettingsScreen() {
  const navigate = useNavigate();
  const { settings, sendMessage } = useDisdriveContext();
  const [days, setDays] = useState<number>(settings?.retention_days || 15);

  const handleSave = () => {
    sendMessage({ 
      action: "update_settings", 
      data: { retention_days: days } 
    });
  };

  return (
    <Container className=" min-vh-100 d-flex align-items-center justify-content-center">
      <Card className="gap-2 p-5 w-75">
        <Row>
          <Col
            onClick={() => navigate("/")}
            style={{
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
            }}
          >
            <ArrowLeftIcon size={24} />
          </Col>
          <Col>
            <h2 className="mb-4">Settings</h2>
          </Col>
        </Row>
        {GetCheckBox("logging", "Enable Logging")}
        <CameraDropDown />
        <NumberSpinner days={days} setDays={setDays} />
        <Button 
          variant="primary" 
          className="mt-3" 
          onClick={handleSave}
        >
          Save Settings
        </Button>
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
