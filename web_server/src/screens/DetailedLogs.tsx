import { useEffect, useState, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Spinner,
  Alert,
} from "react-bootstrap";
import { ArrowLeftIcon } from "@primer/octicons-react";
import jsPDF from "jspdf";

interface LogDetails {
  session_id: number;
  session_start: string;
  session_end: string;
  behavior_id: number;
  behavior: string;
  behavior_time_start: string;
  behavior_time_end: string;
  snapshot: string;
}

function DetailedLogs() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [logDetails, setLogDetails] = useState<LogDetails[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const port = "8767";

  useEffect(() => {
    if (!id) {
      setError("Invalid session ID");
      setIsLoading(false);
      return;
    }

    const ws = new WebSocket(
      `ws://${window.location.hostname}:${port}/ws/logs/${id}`
    );

    ws.onopen = () => {
      console.log(`Connected to WebSocket for session ${id}`);
      ws.send("get_details");
    };

    ws.onmessage = (event) => {
      try {
        const data: LogDetails[] = JSON.parse(event.data);
        if (!data) {
          setError("An error has occured!");
        } else {
          setLogDetails(data);
        }
        setIsLoading(false);
      } catch (error) {
        console.error("Failed to parse session details:", error);
        setError("Failed to parse session details");
      }
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => ws.close(); // Close WebSocket when component unmounts
  }, [id]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const handleDownloadPDF = async () => {
    if (!logDetails || !logDetails.length) {
      alert("No log details available to export");
      return;
    }

    setIsExporting(true);

    try {
      // Create PDF document
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
      });

      // PDF dimensions
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();

      // Add title centered
      pdf.setFontSize(20);
      pdf.setFont("helvetica", "bold");
      pdf.text(
        `Session #${logDetails[0].session_id} Details`,
        pageWidth / 2,
        20,
        { align: "center" }
      );

      let yPosition = 40;

      // Add session information centered
      pdf.setFontSize(12);
      pdf.setFont("helvetica", "normal");
      pdf.text(
        `Session Started: ${formatDate(logDetails[0].session_start)}`,
        pageWidth / 2,
        yPosition,
        { align: "center" }
      );
      yPosition += 8;

      const statusText = logDetails[0].session_end
        ? `Status: Completed (${formatDate(logDetails[0].session_end)})`
        : "Status: Ongoing";
      pdf.text(statusText, pageWidth / 2, yPosition, { align: "center" });
      yPosition += 15;

      // Add behaviors section header centered
      pdf.setFontSize(20);
      pdf.setFont("helvetica", "bold");
      pdf.text("LOGS", pageWidth / 2, yPosition, { align: "center" });
      yPosition += 10;

      if (!logDetails[0].behavior) {
        pdf.setFontSize(12);
        pdf.setFont("helvetica", "italic");
        pdf.text(
          "No behaviors recorded, logging may be off during this session.",
          pageWidth / 2,
          yPosition,
          { align: "center" }
        );
      } else {
        // Adjust yPosition to add lines between pictures on pages with two pictures
        let isFirstPage = true;

        for (let i = 0; i < logDetails.length; i++) {
          const log = logDetails[i];

          // Calculate required space for one behavior and image
          const textHeight = 20; // Approximate height for text block
          const imgWidth = Math.min(pageWidth - 60, 120); // Reduced width
          const imgHeight = log.snapshot ? (120 / 160) * imgWidth : 0; // Maintain aspect ratio
          const requiredHeight = textHeight + imgHeight + 10; // Add padding

          if (isFirstPage) {
            if (yPosition + requiredHeight > pageHeight - 40) {
              pdf.addPage();
              yPosition = 40;
              isFirstPage = false;
            }

            // Add behavior header centered
            pdf.setFontSize(12);
            pdf.setFont("helvetica", "bold");
            pdf.text(`Behavior: ${log.behavior}`, pageWidth / 2, yPosition, {
              align: "center",
            });
            yPosition += 7;

            // Add timing information centered
            pdf.setFontSize(10);
            pdf.setFont("helvetica", "normal");
            pdf.text(
              `Started: ${formatDate(log.behavior_time_start)}`,
              pageWidth / 2,
              yPosition,
              { align: "center" }
            );
            yPosition += 5;
            pdf.text(
              `Ended: ${formatDate(log.behavior_time_end)}`,
              pageWidth / 2,
              yPosition,
              { align: "center" }
            );
            yPosition += 10;

            // Add image if available centered
            if (log.snapshot) {
              try {
                const img = new Image();
                img.src = `data:image/jpeg;base64,${log.snapshot}`;

                await new Promise<void>((resolve, reject) => {
                  img.onload = () => resolve();
                  img.onerror = () => reject(new Error("Image loading failed"));
                  setTimeout(() => resolve(), 1000);
                });

                pdf.addImage(
                  `data:image/jpeg;base64,${log.snapshot}`,
                  "JPEG",
                  (pageWidth - imgWidth) / 2,
                  yPosition,
                  imgWidth,
                  imgHeight,
                  undefined,
                  "FAST"
                );

                yPosition += imgHeight + 15;
              } catch (error) {
                console.error("Failed to add image to PDF:", error);
                pdf.text("[Image could not be added to PDF]", pageWidth / 2, yPosition, {
                  align: "center",
                });
                yPosition += 10;
              }
            }

            isFirstPage = false;
          } else {
            // Handle subsequent pages with two images per page
            const nextLog = logDetails[i + 1];
            const nextImgHeight = nextLog?.snapshot ? (120 / 160) * imgWidth : 0;
            const nextRequiredHeight = textHeight + nextImgHeight + 10;

            if (yPosition + requiredHeight + nextRequiredHeight > pageHeight - 40) {
              pdf.addPage();
              yPosition = 20;
            }

            // Add current behavior and image
            pdf.setFontSize(12);
            pdf.setFont("helvetica", "bold");
            pdf.text(`Behavior: ${log.behavior}`, pageWidth / 2, yPosition, {
              align: "center",
            });
            yPosition += 7;

            pdf.setFontSize(10);
            pdf.setFont("helvetica", "normal");
            pdf.text(
              `Started: ${formatDate(log.behavior_time_start)}`,
              pageWidth / 2,
              yPosition,
              { align: "center" }
            );
            yPosition += 5;
            pdf.text(
              `Ended: ${formatDate(log.behavior_time_end)}`,
              pageWidth / 2,
              yPosition,
              { align: "center" }
            );
            yPosition += 10;

            if (log.snapshot) {
              try {
                const img = new Image();
                img.src = `data:image/jpeg;base64,${log.snapshot}`;

                await new Promise<void>((resolve, reject) => {
                  img.onload = () => resolve();
                  img.onerror = () => reject(new Error("Image loading failed"));
                  setTimeout(() => resolve(), 1000);
                });

                pdf.addImage(
                  `data:image/jpeg;base64,${log.snapshot}`,
                  "JPEG",
                  (pageWidth - imgWidth) / 2,
                  yPosition,
                  imgWidth,
                  imgHeight,
                  undefined,
                  "FAST"
                );

                yPosition += imgHeight + 15;
              } catch (error) {
                console.error("Failed to add image to PDF:", error);
                pdf.text("[Image could not be added to PDF]", pageWidth / 2, yPosition, {
                  align: "center",
                });
                yPosition += 10;
              }
            }

            // Add a little space below the line before the second picture
            pdf.setLineWidth(0.5); // Increase line thickness
            pdf.setDrawColor(200, 200, 200);
            pdf.line(15, yPosition - 5, pageWidth - 15, yPosition - 5);
            yPosition += 5; // Add space below the line

            // Add next behavior and image if available
            if (nextLog) {
              pdf.setFontSize(12);
              pdf.setFont("helvetica", "bold");
              pdf.text(`Behavior: ${nextLog.behavior}`, pageWidth / 2, yPosition, {
                align: "center",
              });
              yPosition += 7;

              pdf.setFontSize(10);
              pdf.setFont("helvetica", "normal");
              pdf.text(
                `Started: ${formatDate(nextLog.behavior_time_start)}`,
                pageWidth / 2,
                yPosition,
                { align: "center" }
              );
              yPosition += 5;
              pdf.text(
                `Ended: ${formatDate(nextLog.behavior_time_end)}`,
                pageWidth / 2,
                yPosition,
                { align: "center" }
              );
              yPosition += 10;

              if (nextLog.snapshot) {
                try {
                  const img = new Image();
                  img.src = `data:image/jpeg;base64,${nextLog.snapshot}`;

                  await new Promise<void>((resolve, reject) => {
                    img.onload = () => resolve();
                    img.onerror = () => reject(new Error("Image loading failed"));
                    setTimeout(() => resolve(), 1000);
                  });

                  pdf.addImage(
                    `data:image/jpeg;base64,${nextLog.snapshot}`,
                    "JPEG",
                    (pageWidth - imgWidth) / 2,
                    yPosition,
                    imgWidth,
                    nextImgHeight,
                    undefined,
                    "FAST"
                  );

                  yPosition += nextImgHeight + 15;
                } catch (error) {
                  console.error("Failed to add image to PDF:", error);
                  pdf.text("[Image could not be added to PDF]", pageWidth / 2, yPosition, {
                    align: "center",
                  });
                  yPosition += 10;
                }
              }

              i++; // Skip the next log as it has been processed
            }
          }
        }
      }

      pdf.setFontSize(12);
      pdf.setTextColor(100, 100, 100);
      pdf.text(
        `Generated: ${new Date().toLocaleString()}`,
        pageWidth / 2,
        pageHeight - 10,
        { align: "center" }
      );

      pdf.save(`Session_${logDetails[0].session_id}_Details.pdf`);

    } catch (error) {
      console.error("PDF generation failed:", error);
      alert("Failed to generate PDF. See console for details.");
    } finally {
      setIsExporting(false);
    }
  };

  if (isLoading) {
    return (
      <Container className="d-flex justify-content-center align-items-center text-center mt-4 min-vh-100">
        <Spinner animation="border" variant="primary" />
        <p>Loading session details...</p>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="mt-4">
        <Alert variant="danger">{error}</Alert>
        <Button variant="secondary" onClick={() => navigate("/logs")}>
          Go Back
        </Button>
      </Container>
    );
  }

  if (!logDetails) {
    return (
      <Container className="mt-4">
        <Alert variant="danger">Session not found</Alert>
        <Button variant="secondary" onClick={() => navigate("/logs")}>
          Go Back
        </Button>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <div ref={contentRef}>
        <Row className="mb-4 align-items-center">
          <Col xs={1}>
            <div onClick={() => navigate("/logs")} style={{ cursor: "pointer" }}>
              <ArrowLeftIcon size={24} />
            </div>
          </Col>
          <Col className="text-center">
            <h2>Session #{logDetails[0].session_id} Details</h2>
          </Col>
        </Row>

        <Card className="mb-3">
          <Card.Body>
            <Card.Title>Session Details</Card.Title>
            <Card.Text>
              <strong>Started:</strong> {formatDate(logDetails[0].session_start)}
              <br />
              <strong>Status:</strong>{" "}
              {logDetails[0].session_end ? (
                <span className="text-success">
                  Completed ({formatDate(logDetails[0].session_end)})
                </span>
              ) : (
                <span className="text-primary">Ongoing</span>
              )}
            </Card.Text>
          </Card.Body>
        </Card>

        <h4 className="fw-bold mt-4">Behaviors</h4>
        {logDetails[0].behavior ? (
          logDetails.map((log, index) => (
            <Card key={index} className="mb-2">
              <Card.Body>
                <Card.Text>
                  <strong>Behavior:</strong> {log.behavior}
                  <br />
                  <img
                    src={`data:image/jpeg;base64,${log.snapshot}`}
                    alt="detected behavior"
                    className="img-fluid mx-auto d-flex"
                  />
                  <br />
                  <strong>Started:</strong> {formatDate(log.behavior_time_start)}
                  <br />
                  <strong>Ended:</strong> {formatDate(log.behavior_time_end)}
                </Card.Text>
              </Card.Body>
            </Card>
          ))
        ) : (
          <p className="text-muted">
            No behaviors recorded, logging may be off during this session.
          </p>
        )}
      </div>

      <Button
        variant="secondary"
        className="mt-3 me-2"
        onClick={() => navigate("/logs")}
      >
        Go Back
      </Button>

      <Button
        variant="primary"
        className="mt-3"
        onClick={handleDownloadPDF}
        disabled={isExporting}
      >
        {isExporting ? (
          <>
            <Spinner animation="border" size="sm" className="me-2" />
            Exporting...
          </>
        ) : (
          "Download PDF"
        )}
      </Button>
    </Container>
  );
}
export default DetailedLogs;