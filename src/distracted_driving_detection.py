"""Runner Software of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel

TRAINING_DATASET_PATH = "./training_dataset"
TESTING_DATASET_PATH = "./testing_dataset"

if __name__ == "__main__":
    obj = DisDriveDataset(TRAINING_DATASET_PATH, HybridModel())
