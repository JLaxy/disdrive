import { Button, Container } from "react-bootstrap";
import { useNavigate } from "react-router";

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
  return (
    <Container className="d-flex flex-column gap-2 min-vh-100 justify-content-center align-items-center">
      {GetHeader()}
      {GetButtons()}
    </Container>
  );
}

export default LandingPage;
