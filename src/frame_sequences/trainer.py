"""Runner Software of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TRAINING_DATASET_PATH = "./training_dataset"
TESTING_DATASET_PATH = "./testing_dataset"

if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()

    dataset = DisDriveDataset(TRAINING_DATASET_PATH, CLIP_LSTM)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for image, prompt in dataloader:
        print(image.shape)
        print(prompt.shape)
        break
