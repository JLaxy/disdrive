import clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

"""Hybrid Model Settings"""
_MODEL = "ViT-B/16"
# Automatically changes to GPU if available
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""LSTM Parameters"""
LSTM_INPUT_SIZE = 512
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2


class HybridModel(nn.Module):
    """The class of the CLIP-LSTM hybrid used for distracted driving detection."""

    def __init__(self):
        """Initializes instance of CLIP-LSTM hybrid model"""
        # REQUIRED; Initializing parent class
        super().__init__()

        # Loading CLIP model
        self.model, self.preprocessor = clip.load(
            _MODEL, device=_DEVICE, jit=False)

        # Freezing default CLIP model
        for param in self.model.parameters():
            param.requires_grad = False

        # Initalizing LSTM Neural Network
        self.lstm = torch.nn.LSTM(
            input_size=LSTM_INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT
        )

    def forward(self, frames: Image, prompts: str):
        """Processes input to the hybrid model to detect distracted driving"""
        clip_result = self.model(frames, prompts)

        # TODO: process by LSTM
        return clip_result

    def preprocess(self, image):
        """Returns image that is preprocessed"""
        return self.preprocessor(image)

    def train():
        """Trains the CLIP-LSTM hybrid model"""
        pass


class DisDriveDataset(Dataset):
    """The class of the dataset which will be used on the Hybrid Model"""

    def __init__(self, dataset_directory, hybrid_model):
        """Initilizes dataset"""
        super().__init__()

        self.dataset_directory = dataset_directory
        self.hybrid_model = hybrid_model  # CLIP and LSTM Hybrid Model

        # Process Dataset
        self.process_dataset(self)

    def __getitem__(self, index):
        pass

    def __len__(self):
        """Returns length of dataset"""

    def process_dataset(self):
        """Creates dataset using supplied dataset directory"""
