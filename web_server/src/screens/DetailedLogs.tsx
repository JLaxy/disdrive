import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { useNavigate, useParams } from "react-router-dom";
import { sampleLogs } from "./LogsScreen"; // Import the logs data

const DetailedLogs = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const selectedLog = sampleLogs.find(log => log.id === Number(id));

  if (!selectedLog) {
    return <div>Log not found</div>;
  }

  return (
    <div className="container mt-4">
      <div className="row">
        <div className="col-md-4">
          <h4 className="fw-bold">DISTRACTION LOGS</h4>
          <div className="list-group">
            {selectedLog.distractions.map((distraction, index) => (
              <div key={index} className="list-group-item">
                {distraction.time} - {distraction.type}
              </div>
            ))}
          </div>
        </div>
        <div className="col-md-8">
          <div className="card p-3">
            <h5 className="fw-bold">SESSION DETAILS</h5>
            <p><strong>Time Started:</strong> {new Date(selectedLog.timestamp).toLocaleTimeString()}</p>
            <p><strong>Time Ended:</strong> {selectedLog.session_end ? 
              new Date(selectedLog.session_end).toLocaleTimeString() : 'Ongoing'}</p>
          </div>
          <button className="btn btn-secondary mt-3" onClick={() => navigate("/logs")}>
            GO BACK
          </button>
        </div>
      </div>
    </div>
  );
};

export default DetailedLogs;
