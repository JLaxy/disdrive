import { Button } from "react-bootstrap";
import LiveFeed from "../components/LiveFeed";
import { Dispatch, SetStateAction, useState } from "react";

function SessionScreen() {
  const [isSessionActive, setIsSessionActive] = useState(false);
  return (
    <div className="d-flex flex-column bg-secondary min-vh-100 container align-items-center justify-content-center">
      <LiveFeed />
      {GetLabels()}
      {ToGetStopButton(isSessionActive, setIsSessionActive)}
    </div>
  );
}

function ToGetStopButton(
  sessionStatus: boolean,
  setIsSessionActive: Dispatch<SetStateAction<boolean>>
) {
  return sessionStatus ? (
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
  );
}

function GetLabels() {
  return (
    <div className="d-flex flex-row w-100">
      <div className="flex-column text-center w-100 justify-content-center align-items-center">
        <p>Status: Not Distracted</p>
        <p>Behavior: Safe Driving</p>
      </div>
      <div className="d-flex w-100 text-center bg-primary justify-content-center align-items-center">
        <p>Time Elapsed: asdasd</p>
      </div>
    </div>
  );
}

export default SessionScreen;
