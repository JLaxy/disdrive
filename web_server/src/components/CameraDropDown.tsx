import React from "react";
import { Dropdown } from "react-bootstrap";
import { useDisdriveContext } from "../contexts/DisdriveContext";

const CameraDropDown: React.FC = () => {
  const { cameras, camera_id, sendMessage } = useDisdriveContext();

  const handleSelect = (eventKey: string | null) => {
    if (eventKey) {
      sendMessage({
        action: "update_camera",
        data: JSON.stringify({ camera_id: eventKey }),
      });
    }
  };

  return (
    <Dropdown onSelect={handleSelect}>
      <Dropdown.Toggle
        variant="secondary"
        id="dropdown-basic"
        className="w-100 text-start justify-content-between d-flex align-items-center"
      >
        {`Camera ${camera_id}`}
      </Dropdown.Toggle>

      <Dropdown.Menu>
        {cameras.map((camera) => (
          <Dropdown.Item eventKey={camera}>Camera {camera}</Dropdown.Item>
        ))}
      </Dropdown.Menu>
    </Dropdown>
  );
};

export default CameraDropDown;
