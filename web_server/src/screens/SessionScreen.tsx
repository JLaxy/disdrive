import { Button } from "react-bootstrap";
import LiveFeed from "../components/LiveFeed";
import { useNavigate } from "react-router";
import { useDisdriveContext } from "../contexts/DisdriveContext";

function SessionScreen() {
  // Retrieve context
  const { hasOngoingSession, setHasOngoingSession } = useDisdriveContext();
  return (
    <div className="d-flex flex-column min-vh-100 bg-dark container align-items-center justify-content-center gap-3">
      <LiveFeed />
      {GetButtons(hasOngoingSession, setHasOngoingSession)}
    </div>
  );
}

function GetButtons(
  hasOngoingSession: boolean,
  setHasOngoingSession: (arg0: boolean) => void
) {
  const navigate = useNavigate();
  return (
    // Go Back Button
    <div className="d-flex flex-row w-100 gap-3">
      <Button
        variant="secondary"
        className="w-100 btn-lg"
        onClick={() => navigate("/")}
      >
        Go Back
      </Button>
      {hasOngoingSession ? (
        <Button
          variant="danger"
          className="btn-lg w-100"
          onClick={() => setHasOngoingSession(false)}
        >
          Stop Session
        </Button>
      ) : (
        <Button
          variant="success"
          className="btn-lg w-100"
          onClick={() => setHasOngoingSession(true)}
        >
          Start Session
        </Button>
      )}
    </div>
  );
}

export default SessionScreen;
