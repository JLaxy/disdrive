"""Runner Software of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import HybridModel, DisDriveDataset
from torch.utils.data import DataLoader
import torch

TRAINING_DATASET_PATH = "./datasets/frame_by_frame/train"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPOCHS = 15  # Number of Epochs
_LEARNING_RATE = 0.0001  # Learning rate for optimizer in training
_WEIGHT_DECAY = 0.00001  # Weight decay for optimizer in training
_TRAINED_MODEL_SAVE_PATH = "./saved_models"
_TO_PREPROCESS_DATA = True

if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()
    CLIP_LSTM.to(_DEVICE)

    dataset = DisDriveDataset(TRAINING_DATASET_PATH,
                              CLIP_LSTM, _TO_PREPROCESS_DATA)

    print(f"sample: {dataset[28][0]}, {len(dataset[28][1])}")

    # dataloader = DataLoader(dataset, batch_size=32,
    #                         shuffle=True, pin_memory=True)

    # for behavior, image in dataloader:
    #     print(behavior)
    #     print(image)
    #     break
