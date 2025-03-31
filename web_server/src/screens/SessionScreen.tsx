import { Button } from "react-bootstrap";
import LiveFeed from "../components/LiveFeed";
import { useNavigate } from "react-router";
import { useDisdriveContext } from "../contexts/DisdriveContext";
import { closeWebSocket } from "../utils/LiveFeedSocketService";

function SessionScreen() {
  // Retrieve context
  const { has_ongoing_session, sendMessage } = useDisdriveContext();
  return (
    <div className="d-flex flex-column min-vh-100 bg-dark container align-items-center justify-content-center gap-3">
      <LiveFeed />
      {GetButtons(has_ongoing_session, sendMessage)}
    </div>
  );
}

function GetButtons(
  hasOngoingSession: boolean,
  sendMessage: (arg0: Record<string, string>) => void
) {
  const navigate = useNavigate();
  return (
    // Go Back Button
    <div className="d-flex flex-row w-100 gap-3">
      <Button
        variant="secondary"
        className="w-100 btn-lg"
        onClick={() => {
          navigate("/");
          closeWebSocket();
        }}
      >
        Go Back
      </Button>
      {hasOngoingSession ? (
        <Button
          variant="danger"
          className="btn-lg w-100"
          // Send message to backend to stop the session
          onClick={() => sendMessage({ action: "stop_session" })}
        >
          Stop Session
        </Button>
      ) : (
        <Button
          variant="success"
          className="btn-lg w-100"
          // Send message to backend to start the session
          onClick={() => sendMessage({ action: "start_session" })}
        >
          Start Session
        </Button>
      )}
    </div>
  );
}

export default SessionScreen;
