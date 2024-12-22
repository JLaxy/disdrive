import clip
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image

"""Hybrid Model Settings"""
_MODEL = "ViT-B/16"  # 224x224
# Automatically changes to GPU if available
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""LSTM Parameters"""
_LSTM_INPUT_SIZE = 512
_LSTM_HIDDEN_SIZE = 512
_LSTM_NUM_LAYERS = 2
_LSTM_DROPOUT = 0.2

# Prompts for distracted driving behaviors
_DRIVING_CATEGORIES_PROMPTS = [
    "a photo of a person driving safely with both hands on the steering wheel",  # c0
    "a photo of a person texting using their mobile phone while driving",        # c1
    "a photo of a person talking on the phone while driving",                    # c2
    "a photo of a person operating the car radio while driving",                 # c3
    "a photo of a person drinking while driving",                                # c4
    "a photo of a person reaching for an object behind them while driving",      # c5
    "a photo of a person with their head down while driving"                     # c6
]

# Descriptions for each category
_CATEGORY_PROMPTS = {
    "c0": _DRIVING_CATEGORIES_PROMPTS[0],
    "c1": _DRIVING_CATEGORIES_PROMPTS[1],
    "c1.1": _DRIVING_CATEGORIES_PROMPTS[1],
    "c2": _DRIVING_CATEGORIES_PROMPTS[2],
    "c2.1": _DRIVING_CATEGORIES_PROMPTS[2],
    "c3": _DRIVING_CATEGORIES_PROMPTS[3],
    "c4": _DRIVING_CATEGORIES_PROMPTS[4],
    "c5": _DRIVING_CATEGORIES_PROMPTS[5],
    "c6": _DRIVING_CATEGORIES_PROMPTS[6]
}


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

        # Freezing default CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        print("Loading LSTM model...")

        # Initalizing LSTM Neural Network
        self.lstm = torch.nn.LSTM(
            input_size=_LSTM_INPUT_SIZE,
            hidden_size=_LSTM_HIDDEN_SIZE,
            num_layers=_LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=_LSTM_DROPOUT
        )

    def forward(self, frames: Image, prompts: str):
        """Processes input to the hybrid model to detect distracted driving"""
        clip_result = self.model(frames, prompts)

        print("forwarded!")
        # TODO: process by LSTM
        return clip_result

    def train():
        """Trains the CLIP-LSTM hybrid model"""
        pass


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
        (image, prompt) = self.dataset_data[index] # Retrieve image and prompt

        # Preprocess then put to device
        image = self.hybrid_model.preprocessor(image).to(_DEVICE)
        prompt = clip.tokenize(prompt).squeeze(0).to(_DEVICE)

        return image, prompt

    def __len__(self):
        """Returns length of dataset"""
        return len(self.dataset_data)

    def process_dataset(self):
        """Creates dataset using supplied dataset directory"""

        print("Building Dataset...")
        # For every Folder
        for category in os.listdir(self.dataset_directory):
            print(f"Category: {category}")
            # Append OS path to folder name
            category_path = os.path.join(self.dataset_directory, category)

            # If file is a folder directory
            if os.path.isdir(category_path):
                # Get equivalent prompt
                prompt = _CATEGORY_PROMPTS.get(category)
                # For each image in folder
                for image in os.listdir(category_path):
                    # Get concatenated image path
                    path = os.path.join(category_path, image)

                    # Add Image and Prompt pair to dataset data
                    self.dataset_data.append((Image.open(path), prompt))
