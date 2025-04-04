import { Button } from "react-bootstrap";

interface NumberSpinnerProps {
  days: number;
  setDays: (value: number) => void;
}

function NumberSpinner({ days, setDays }: NumberSpinnerProps) {
  const handleClick = (operation: string) => {
    if (operation == "add" && days < 30) setDays(days + 1);
    else if (operation == "sub" && days > 15) setDays(days - 1);
  };

  return (
    <div className="d-flex flex-row">
      <Button
        variant="primary"
        className="btn-lrg"
        onClick={() => handleClick("sub")}
      >
        -
      </Button>
      <div className="align-items-center d-flex">{days}</div>
      <Button
        className="btn-lrg"
        variant="primary"
        onClick={() => handleClick("add")}
      >
        +
      </Button>
    </div>
  );
}

export default NumberSpinner;
