import clip
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
import PIL

"""Hybrid Model Settings"""
_MODEL = "ViT-B/16"  # 224x224
# Automatically changes to GPU if available
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Number of Distracted Driving Behaviors"""
_NUM_OF_CLASSES = 7

"""LSTM Parameters"""
_LSTM_INPUT_SIZE = 512
_LSTM_HIDDEN_SIZE = 256
_LSTM_NUM_LAYERS = 6


class HybridModel(nn.Module):
    """The class of the CLIP-LSTM hybrid used for distracted driving detection."""

    def __init__(self):
        """Initializes instance of CLIP-LSTM hybrid model"""
        # REQUIRED; Initializing parent class
        super().__init__()

        print("Loading CLIP model...")

        # Loading CLIP model
        self.clip_model, self.preprocessor = clip.load(
            _MODEL, device=_DEVICE, jit=False)

        print("Loading LSTM model...")

        # Initalizing LSTM Neural Network
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            input_size=_LSTM_INPUT_SIZE,
            hidden_size=_LSTM_HIDDEN_SIZE,
            num_layers=_LSTM_NUM_LAYERS,
            batch_first=True,
        )

        self.fc = nn.Linear(_LSTM_HIDDEN_SIZE, _NUM_OF_CLASSES)

    def forward(self, frames: list[PIL.Image]):
        """Processes input to the hybrid model to detect distracted driving"""
        vectorized_frames: list[torch.Tensor] = list()
        for image in frames:  # Iterate through each and every image
            vectorized_frames.append(self.clip_model)

        print("forwarded!")
        # TODO: process by LSTM
        return clip_result


class DisDriveDataset(Dataset):
    """The class of the dataset which will be used on the Hybrid Model"""

    def __init__(self, dataset_directory: str, hybrid_model: HybridModel):
        """Initilizes dataset"""
        super().__init__()

        self.dataset_directory = dataset_directory  # Filepath of Dataset
        self.hybrid_model = hybrid_model  # CLIP and LSTM Hybrid Model

        # List of Image-Text Pairs in Tensor form
        self.dataset_data = []

        # Process Dataset
        self.process_dataset()

    def __getitem__(self, index):
        """Returns Dataset Data at specific index"""
        (image,
         category) = self.dataset_data[index]  # Retrieve image and prompt

        # Preprocess then put to device
        image = self.hybrid_model.preprocessor(image).to(_DEVICE)

        return image, category

    def __len__(self):
        """Returns length of dataset"""
        return len(self.dataset_data)
    
    def process_dataset():
        pass
