import React, { useState } from "react";
import { Dropdown, Modal, Button } from "react-bootstrap";
import { useDisdriveContext } from "../contexts/DisdriveContext";

const ViewDropDown: React.FC = () => {
  const { camera_view, sendMessage } = useDisdriveContext();
  const [showConfirm, setShowConfirm] = useState(false);
  const [selectedView, setSelectedView] = useState<string | null>(null);

  const handleSelect = (eventKey: string | null) => {
    if (eventKey) {
      setSelectedView(eventKey);
      setShowConfirm(true); // Show confirmation modal
    }
  };

  const handleConfirm = () => {
    if (selectedView) {
      // Determine the model path based on the selected view
      const modelPath =
        selectedView === "Front"
          ? "./saved_models/front_model.pth"
          : "./saved_models/side_model.pth";

      // Send the new camera view and model path to the backend
      sendMessage({
        action: "update_camera_view",
        data: JSON.stringify({
          camera_view: selectedView,
          model_path: modelPath,
        }),
      });
    }
    setShowConfirm(false); // Close the modal
  };

  const handleCancel = () => {
    setSelectedView(null); // Reset selected view
    setShowConfirm(false); // Close the modal
  };

  return (
    <>
      <Dropdown onSelect={handleSelect}>
        <Dropdown.Toggle
          variant="secondary"
          id="dropdown-basic"
          className="w-100 text-start justify-content-between d-flex align-items-center"
        >
          {`View: ${camera_view}`}
        </Dropdown.Toggle>

        <Dropdown.Menu>
          <Dropdown.Item eventKey="Front">Front View</Dropdown.Item>
          <Dropdown.Item eventKey="Side">Side View</Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown>

      {/* Confirmation Modal */}
      <Modal show={showConfirm} onHide={handleCancel} centered>
        <Modal.Header closeButton>
          <Modal.Title>Confirm Camera Angle</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to switch to {selectedView} View?
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleCancel}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleConfirm}>
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default ViewDropDown;
