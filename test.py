import cv2
import time


def open_camera(frame_rate=30):
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_delay = 1 / frame_rate  # Calculate time delay between frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

        time.sleep(frame_delay)  # Control frame rate

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    X = 10  # Set the desired frame rate (frames per second)
    open_camera(X)
