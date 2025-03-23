from backend.model_loader import ModelLoader


class Disdrive:
    def __init__(self):
        print("Starting Disdrive...")

        # Load Model
        self.hybrid_model = ModelLoader()


# Main function
if __name__ == "__main__":
    disdrive = Disdrive()
