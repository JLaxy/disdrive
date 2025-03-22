import React, { useState } from "react";
import { Dropdown } from "react-bootstrap";

const CameraDropDown: React.FC = () => {
  const [selected, setSelected] = useState<string>("Select an option");

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
        <Dropdown.Item eventKey="Camera 1">Camera 1</Dropdown.Item>
        <Dropdown.Item eventKey="Camera 2">Camera 2</Dropdown.Item>
        <Dropdown.Item eventKey="Camera 3">Camera 3</Dropdown.Item>
      </Dropdown.Menu>
    </Dropdown>
  );
};

export default CameraDropDown;
