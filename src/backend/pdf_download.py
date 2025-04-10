# import asyncio
# import json
# from fastapi import WebSocket
# from fpdf import FPDF

# async def handle_pdf_request(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             message = await websocket.receive_text()
#             data = json.loads(message)

#             if data.get("action") == "generate_pdf":
#                 session_details = data.get("session_details")
#                 if not session_details:
#                     await websocket.send_text(json.dumps({"status": "error", "message": "No session details provided"}))
#                     continue

#                 # Generate PDF
#                 pdf = FPDF()
#                 pdf.add_page()
#                 pdf.set_font("Arial", size=12)
#                 pdf.cell(200, 10, txt=f"Session #{session_details['session_id']} Details", ln=True, align="C")
#                 pdf.ln(10)

#                 pdf.cell(200, 10, txt=f"Started: {session_details['session_start']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Ended: {session_details['session_end'] or 'Ongoing'}", ln=True)
#                 pdf.ln(10)

#                 pdf.cell(200, 10, txt="Behaviors:", ln=True)
#                 for behavior in session_details.get("behaviors", []):
#                     pdf.cell(200, 10, txt=f"- {behavior['behavior']} (Start: {behavior['behavior_time_start']}, End: {behavior['behavior_time_end']})", ln=True)

#                 # Save PDF to a buffer
#                 pdf_output = f"session_{session_details['session_id']}.pdf"
#                 pdf.output(pdf_output)

#                 # Send PDF back to client
#                 with open(pdf_output, "rb") as pdf_file:
#                     pdf_bytes = pdf_file.read()
#                     await websocket.send_bytes(pdf_bytes)

#     except Exception as e:
#         print(f"Error in handle_pdf_request: {e}")
#     finally:
#         await websocket.close()