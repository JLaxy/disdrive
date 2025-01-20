"""Runner Software of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader

TRAINING_DATASET_PATH = "./datasets/frame_sequences/train"

if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()

    dataset = DisDriveDataset(TRAINING_DATASET_PATH, CLIP_LSTM)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for behavior, image in dataloader:
        print(behavior)
        print(image)
        break
