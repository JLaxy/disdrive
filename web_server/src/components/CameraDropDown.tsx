import React, { useState } from "react";
import { Dropdown } from "react-bootstrap";
import { useDisdriveContext } from "../contexts/DisdriveContext";

const CameraDropDown: React.FC = () => {
  const [selected, setSelected] = useState<string>("Select an option");
  const { cameras } = useDisdriveContext();

  const handleSelect = (eventKey: string | null) => {
    if (eventKey) {
      setSelected(eventKey);
    }
  };

  return (
    <Dropdown onSelect={handleSelect}>
      <Dropdown.Toggle
        variant="primary"
        id="dropdown-basic"
        className="w-100 text-start justify-content-between d-flex align-items-center"
      >
        {selected}
      </Dropdown.Toggle>

      <Dropdown.Menu>
        {cameras.map((camera) => (
          <Dropdown.Item eventKey={`Camera ${camera}`}>
            Camera {camera}
          </Dropdown.Item>
        ))}
      </Dropdown.Menu>
    </Dropdown>
  );
};

export default CameraDropDown;
