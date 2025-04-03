import { Button, Container } from "react-bootstrap";
import { useNavigate } from "react-router";
import { useEffect } from "react";
import { useDisdriveContext } from "../contexts/DisdriveContext";

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
  const { sendMessage } = useDisdriveContext();

  const handleShutdown = () => {
    console.log("Initiating shutdown via button...");
    sendMessage({ action: "shutdown_system" });
  };

  return (
    <div className="d-flex flex-column gap-3 w-100 justify-content-center align-items-center">
      {GetButton("primary", "View Session", "/session")}
      {GetButton("primary", "View Logs", "/logs")}
      {GetButton("primary", "Settings", "/settings")}
      <Button
        variant="danger"
        className="btn-lg w-75"
        onClick={handleShutdown}
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
  const { sendMessage } = useDisdriveContext();

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === "C" || event.key === "c") {
        console.log("Shutdown key pressed");
        sendMessage({ action: "shutdown_system" });
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [sendMessage]);

  return (
    <Container className="d-flex flex-column gap-2 min-vh-100 justify-content-center align-items-center">
      {GetHeader()}
      {GetButtons()}
    </Container>
  );
}

export default LandingPage;
