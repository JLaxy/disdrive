import { Button } from "react-bootstrap";
import { useDisdriveContext } from "../contexts/DisdriveContext";

function NumberSpinner() {
  const {retention_days, sendMessage} = useDisdriveContext();

  const handleClick = (operation: string) => {
    if (operation == "add" && retention_days < 30) sendMessage({ action: "update_retention_days", data: {retention_days: retention_days + 1} });
    else if (operation == "sub" && retention_days > 15) sendMessage({ action: "update_retention_days", data: {retention_days: retention_days - 1} });
  };

  return (
    <div className="d-flex flex-row align-items-center justify-content-between">
      <div className="fw-semibold align-items-center">Log Retention Days:</div>
      <div className="d-flex flex-row w-75 justify-content-between">
      <Button
        variant="primary"
        className="btn-lrg col-1 justify-content-center d-flex"
        onClick={() => handleClick("sub")}
      >
        -
      </Button>
      <div className="align-items-center d-flex fw-bold">{retention_days}</div>
      <Button
        className="btn-lrg col-1 justify-content-center d-flex"
        variant="primary"
        onClick={() => handleClick("add")}
      >
        +
      </Button>
      </div>
    </div>
  );
}

export default NumberSpinner;
