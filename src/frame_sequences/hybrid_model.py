import clip
import numpy
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import PIL

"""Hybrid Model Settings"""
_MODEL = "ViT-B/16"  # 224x224
# Automatically changes to GPU if available
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Number of Distracted Driving Behaviors"""
_NUM_OF_CLASSES = 9

"""LSTM Parameters"""
_LSTM_INPUT_SIZE = 512
_LSTM_HIDDEN_SIZE = 256
_LSTM_NUM_LAYERS = 6

_BEHAVIOR_LABEL = {
    "a": 0,  # Safe Driving
    "b": 1,  # Texting Right
    "c": 2,  # Texting Left
    "d": 3,  # Talking using Phone Right
    "e": 4,  # Talking using Phone Left
    "f": 5,  # Operating Radio
    "g": 6,  # Drinking
    "h": 7,  # Head Down
    "i": 8   # Look Behind
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

        print("Loading LSTM model...")

        # Initalizing LSTM Neural Network
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            input_size=_LSTM_INPUT_SIZE,
            hidden_size=_LSTM_HIDDEN_SIZE,
            num_layers=_LSTM_NUM_LAYERS,
            batch_first=True,
            device=_DEVICE
        )

        self.fc = nn.Linear(_LSTM_HIDDEN_SIZE, _NUM_OF_CLASSES, device=_DEVICE)

        print(f"Successfully Loaded! Using device: {_DEVICE}")

    def forward(self, tensor_sequence):
        """Processes input to the hybrid model to detect distracted driving"""

        lstm_output, (h_n, c_n) = self.lstm(
            tensor_sequence)  # LSTM Forward Pass
        last_state = lstm_output[:, -1, :]
        output = self.fc(last_state)

        return output


class DisDriveDataset(Dataset):
    """The class of the dataset which will be used on the Hybrid Model"""

    def __init__(self, dataset_directory: str, hybrid_model: HybridModel):
        """Initilizes dataset"""
        print("Creating dataset...")

        super().__init__()

        self.dataset_directory = dataset_directory  # Filepath of Dataset
        self.hybrid_model = hybrid_model  # CLIP and LSTM Hybrid Model

        # List of Image-Text Pairs in Tensor form
        self.dataset_data = []  # [(behavior, [IMAGE_SEQUENCE])]

        # Process Dataset
        self.__process_dataset()

    def __getitem__(self, index):
        """Returns Dataset Data at specific index"""
        (behavior,
         frame_paths) = self.dataset_data[index]  # Retrieve image and prompt

        frames = []  # List containing features of frames

        for frame_path in frame_paths:  # For every frame in list of paths
            preprocessed = self.hybrid_model.preprocessor(
                # Preprocess image
                Image.open(frame_path)).unsqueeze(0).to(_DEVICE)

            # Extract features using CLIP
            with torch.no_grad():
                features = self.hybrid_model.clip_model.encode_image(
                    preprocessed)

            # Remove first dimension then add features to list
            frames.append(features.squeeze(0).cpu().numpy())

        return behavior, numpy.array(frames)

    def __len__(self):
        """Returns length of dataset"""
        return len(self.dataset_data)

    def __process_dataset(self):
        """Read dataset data"""

        print("Processing dataset...")

        # Iterate through each behavior
        for behavior_folder in os.listdir(self.dataset_directory):

            behavior = _BEHAVIOR_LABEL.get(
                behavior_folder)  # Get type of behavior

            # Sets current behavior folder
            behavior_path = os.path.join(
                self.dataset_directory, behavior_folder)

            # If current path is a directory; contains folders
            if os.path.isdir(behavior_path):
                # For every grouped sequence in current behavior folder
                for sequence_folder in os.listdir(behavior_path):

                    # print(f"Sequence: {behavior}, {sequence_folder}")

                    # Folder of current behavior sequence
                    sequence_path = os.path.join(
                        behavior_path, sequence_folder)
                    frame_list = []  # List of frames in a sequence of behavior

                    # For every Frame in Sequence Folder
                    for frame in os.listdir(sequence_path):
                        frame_path = os.path.join(sequence_path, frame)
                        # Add Frame to List
                        frame_list.append(frame_path)

                    self.dataset_data.append(
                        (behavior, frame_list))  # Add to dataset_data

        print(f"Total number of processed sequences: {len(self.dataset_data)}")
