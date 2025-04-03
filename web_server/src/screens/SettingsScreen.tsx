import { Button } from "react-bootstrap";
import LiveFeed from "../components/LiveFeed";
import { useNavigate } from "react-router";
import { useDisdriveContext } from "../contexts/DisdriveContext";
import { closeWebSocket } from "../utils/LiveFeedSocketService";
import { useEffect } from "react";

function SessionScreen() {
  const { has_ongoing_session, sendMessage } = useDisdriveContext();

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === "A" || event.key === "a") {
        console.log("Start Session key pressed");
        if (!has_ongoing_session) {
          sendMessage({ action: "start_session" });
        }
      } else if (event.key === "B" || event.key === "b") {
        console.log("Stop Session key pressed");
        if (has_ongoing_session) {
          sendMessage({ action: "stop_session" });
        }
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [has_ongoing_session, sendMessage]);

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

  const handleSessionToggle = () => {
    const action = hasOngoingSession ? "stop_session" : "start_session";
    console.log(`Sending ${action} command...`);
    sendMessage({ action });
  };

  return (
    <div className="d-flex flex-row w-100 gap-3">
      <Button
        variant="secondary"
        className="w-100 btn-lg"
        onClick={() => {
          closeWebSocket();
          navigate("/");
        }}
      >
        Go Back
      </Button>
      <Button
        variant={hasOngoingSession ? "danger" : "success"}
        className="btn-lg w-100"
        onClick={handleSessionToggle}
      >
        {hasOngoingSession ? "Stop Session" : "Start Session"}
      </Button>
    </div>
  );
}

export default SessionScreen;
