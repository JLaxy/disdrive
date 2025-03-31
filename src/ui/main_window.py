import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


class MainWindow(QMainWindow):
    """QMainWindow Class of DisDrive"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("DisDrive: Distracted Driving Detection")
        self.setGeometry(0, 0, 1366, 720)  # X, Y, Width, Height


if __name__ == "__main__":
    print("Starting Application...")

    app = QApplication(sys.argv)  # Create app then pass args
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
