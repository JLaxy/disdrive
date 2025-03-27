import { Button, Card, Col, Container, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router";
import CameraDropDown from "../components/CameraDropDown";
import { useState, useEffect } from "react";

function SettingsScreen() {
  const navigate = useNavigate();
  const [toDistractAlarm, setDistractAlarm] = useState(false);
  const [toLogging, setLogging] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState("Front View");

  // Load saved settings
  useEffect(() => {
    const savedSettings = localStorage.getItem("settings");
    if (savedSettings) {
      const { toDistractAlarm, toLogging, camera } = JSON.parse(savedSettings);
      setDistractAlarm(toDistractAlarm ?? false);
      setLogging(toLogging ?? false);
      setSelectedCamera(camera ?? "Front View");
    }
  }, []);

  const handleSave = () => {
    const settings = {
      toDistractAlarm,
      toLogging,
      camera: selectedCamera
    };
    localStorage.setItem("settings", JSON.stringify(settings));
    navigate("/");
  };

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
        <CameraDropDown 
          selected={selectedCamera}
          onSelect={setSelectedCamera}
        />
        {GetButtons(handleSave)}
      </Card>
    </Container>
  );

  function GetCheckBox(id: string, text: string, checked: boolean, onChange: (e: React.ChangeEvent<HTMLInputElement>) => void) {
    return (
      <Form.Check type="checkbox" id={id}>
        <Form.Check.Input type="checkbox" checked={checked} onChange={onChange} />
        <Form.Check.Label className="fw-semibold">{text}</Form.Check.Label>
      </Form.Check>
    );
  }

  function GetButtons(onSave: () => void) {
    return (
      <Row className="g-2 pt-3">
        <Col>{GetButton("danger", "Go Back", () => navigate("/"))}</Col>
        <Col>{GetButton("success", "Save", onSave)}</Col>
      </Row>
    );
  }

  function GetButton(variant: string, text: string, onClick: () => void) {
    return (
      <Button variant={variant} className="w-100" onClick={onClick}>
        {text}
      </Button>
    );
  }
}

export default SettingsScreen;
