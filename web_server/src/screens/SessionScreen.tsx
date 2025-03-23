import { Button } from "react-bootstrap";
import LiveFeed from "../components/LiveFeed";
import { Dispatch, SetStateAction, useState } from "react";
import { useNavigate } from "react-router";

function SessionScreen() {
  const [isSessionActive, setIsSessionActive] = useState(false);
  return (
    <div className="d-flex flex-column min-vh-100 bg-dark container align-items-center justify-content-center gap-3">
      <LiveFeed />
      {GetButtons(isSessionActive, setIsSessionActive)}
    </div>
  );
}

function GetButtons(
  sessionStatus: boolean,
  setIsSessionActive: Dispatch<SetStateAction<boolean>>
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
      {sessionStatus ? (
        <Button
          variant="danger"
          className="btn-lg w-100"
          onClick={() => setIsSessionActive(false)}
        >
          Stop Session
        </Button>
      ) : (
        <Button
          variant="success"
          className="btn-lg w-100"
          onClick={() => setIsSessionActive(true)}
        >
          Start Session
        </Button>
      )}
    </div>
  );
}

export default SessionScreen;
