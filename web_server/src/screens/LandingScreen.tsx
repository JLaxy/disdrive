import { Button, Container, Form, Row, Col } from "react-bootstrap";
import { useNavigate } from "react-router";
import { useState, useEffect } from "react";

const DarkModeToggle = () => {
  // State to track the dark mode status
  const storedTheme = localStorage.getItem("theme");
  const initialTheme = storedTheme ? storedTheme : "light";

  const [isDarkMode, setIsDarkMode] = useState(initialTheme === "dark");

  // Effect to apply dark mode to the body when the state changes
  useEffect(() => {
    document.body.dataset.bsTheme = isDarkMode ? "dark" : "light";
    localStorage.setItem("theme", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  return (
    <Container className="p-3 d-flex justify-content-end">
      <Row>
        <Col xs="auto">
          <div>
            <Form.Check
              type="switch"
              id="custom-switch"
              label="Dark Mode"
              checked={isDarkMode}
              onChange={() => setIsDarkMode(!isDarkMode)} // Toggle dark mode
            />
          </div>
        </Col>
      </Row>
    </Container>
  );
};

function GetButton(btnVariant: string, btnText: string, btnNavigate: string) {
  const navigate = useNavigate();
  return (
    <Button
      variant={btnVariant}
      className="btn-lg w-75"
      onClick={() => navigate(btnNavigate)}
    >
      {btnText}
    </Button>
  );
}

function GetButtons() {
  return (
    <div className="d-flex flex-column gap-3 w-100 justify-content-center align-items-center">
      {GetButton("primary", "View Session", "/session")}
      {GetButton("primary", "View Logs", "/logs")}
      {GetButton("primary", "Settings", "/settings")}
      <Button
        variant="danger"
        className="btn-lg w-75"
        onClick={() => alert("shutting down!")}
      >
        Shutdown
      </Button>
    </div>
  );
}

function GetHeader() {
  return (
    <div className="mb-5 text-center p-5">
      <h2>DisDrive: Distracted Driving Detection</h2>
    </div>
  );
}

function LandingPage() {
  // Ensure the theme is applied on initial load
  useEffect(() => {
    const storedTheme = localStorage.getItem("theme") || "light";
    document.body.dataset.bsTheme = storedTheme;
  }, []);

  return (
    <Container className="d-flex flex-column gap-2 min-vh-100 justify-content-center align-items-center">
      {DarkModeToggle()}
      {GetHeader()}
      {GetButtons()}
    </Container>
  );
}

export default LandingPage;
