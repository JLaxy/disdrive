import { Card, Col, Container, Dropdown, Form, Row } from "react-bootstrap";
import { useNavigate } from "react-router";
import CameraDropDown from "../components/CameraDropDown";
import { useDisdriveContext } from "../contexts/DisdriveContext";
import { ArrowLeftIcon } from "@primer/octicons-react";
import NumberSpinner from "../components/NumberSpinner";

function SettingsScreen() {
  const navigate = useNavigate();

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
        {GetViewSelector()}
        <NumberSpinner />
      </Card>
    </Container>
  );
}

function GetViewSelector() {
  const { sendMessage, current_view } = useDisdriveContext();

  const handleSelect = (eventKey: string | null) => {
    if (eventKey && current_view.toString() != eventKey) {
      sendMessage({
        action: "update_camera_view",
        data: JSON.stringify({ selected_view: eventKey }),
      });
    }
  };

  return (
    <div className="d-flex flex-column">
      <p className="mb-0 fw-semibold">Selected View</p>
      <Dropdown onSelect={handleSelect}>
        <Dropdown.Toggle
          variant="secondary"
          id="dropdown-basic"
          className="w-100 text-start justify-content-between d-flex align-items-center"
        >
          {current_view === "front" ? "Front" : "Side"}
        </Dropdown.Toggle>
        <Dropdown.Menu>
          <Dropdown.Item eventKey={"front"}>Front</Dropdown.Item>
          <Dropdown.Item eventKey={"side"}>Side</Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown>
    </div>
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
