import clip
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy

"""Hybrid Model Settings"""
_MODEL = "ViT-B/16"  # 224x224
# Automatically changes to GPU if available
_DEVICE = "cpu" if torch.cuda.is_available() else "cpu"

"""LSTM Parameters"""
_LSTM_INPUT_SIZE = 6
_LSTM_HIDDEN_SIZE = 512
_LSTM_NUM_LAYERS = 2
_LSTM_DROPOUT = 0.3

_NUM_OF_CLASSES = 6

# Prompts for distracted driving behaviors
_DRIVING_CATEGORIES_PROMPTS = [
    "a photo of a person driving safely with both hands on the steering wheel",
    "a photo of a person texting using their mobile phone while driving",
    "a photo of a person talking on the phone while driving",
    "a photo of a person drinking while driving",
    "a photo of a person with their head down while driving",                              # c4
    "a photo of a person looking behind while driving",
]

_BEHAVIOR_LABEL = {
    "a": 0,  # Safe Driving
    "b": 1,  # Texting Right
    "c": 1,  # Texting Left
    "d": 2,  # Talking using Phone Right
    "e": 2,  # Talking using Phone Left
    "f": 3,  # Drinking
    "g": 4,  # Head Down
    "h": 5,  # Look Behind
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
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            input_size=_LSTM_INPUT_SIZE,
            hidden_size=_LSTM_HIDDEN_SIZE,
            num_layers=_LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=_LSTM_DROPOUT
        )

        self.fc = nn.Linear(_LSTM_HIDDEN_SIZE, _NUM_OF_CLASSES, device=_DEVICE)

        print("Precomputing prompts...")
        self.precompute_prompts()

    def precompute_prompts(self):
        """Precomputes prompts for LSTM model"""
        # Tokenizes prompts for LSTM model
        self.prompts = clip.tokenize(_DRIVING_CATEGORIES_PROMPTS).to(_DEVICE)

        with torch.no_grad():
            # Encodes prompts
            self.prompts = self.clip_model.encode_text(self.prompts)
            # Normalize
            self.prompts /= self.prompts.norm(dim=-1, keepdim=True)

    def preprocess(self, save_directory: str, frame_path: str, frame_name: str):
        """Preprocess image, compute logits then save to disk"""

        # If temp folder for feature does not exist
        if not os.path.exists(save_directory):
            # Create temp folder
            os.makedirs(save_directory)

        preprocessed = self.preprocessor(Image.open(frame_path)).unsqueeze(
            0).to(_DEVICE)  # Open image, preprocess then save to device

        with torch.no_grad():
            features = self.clip_model.encode_image(
                preprocessed)  # Extract features
            # Normalize features using L2 norm
            features = features / features.norm(dim=-1, keepdim=True)
            # Get logits, edit dimension then convert into numpy
            logits = (100.0 * features @
                      self.prompts.T).squeeze(0).cpu().numpy()

        numpy.save(os.path.join(save_directory, frame_name.replace(".jpg", "")),
                   logits)  # Save logits to disk

    def forward(self, prediction_sequence):
        """Processes input to the hybrid model to detect distracted driving"""

        lstm_output, (h_n, c_n) = self.lstm(
            prediction_sequence)  # LSTM Forward Pass

        last_state = lstm_output[:, -1, :]
        output = self.fc(last_state)

        return output


class DisDriveDataset(Dataset):
    """The class of the dataset which will be used on the Hybrid Model"""

    def __init__(self, dataset_directory: str, hybrid_model: HybridModel, to_get_features=True):
        """Initilizes dataset"""
        print("Creating dataset...")

        super().__init__()

        self.to_get_features = to_get_features
        self.dataset_directory = dataset_directory  # Filepath of Dataset
        self.hybrid_model = hybrid_model  # CLIP and LSTM Hybrid Model

        # List of Image-Text Pairs in Tensor form
        self.dataset_data = []  # [(behavior, [PREDICTION_LOGITS])]

        # Process Dataset
        self.__process_dataset()

    def __getitem__(self, index):
        """Returns Dataset Data at specific index"""
        # Retrieve behavior and logits
        (behavior, logits_path) = self.dataset_data[index]

        logits = []  # List containing logit of frames

        # For every logit in logit_path path
        for logits_file in os.listdir(logits_path):
            # Create path of logit
            path = os.path.join(
                logits_path, logits_file)

            # Load logit from disk
            logit = numpy.load(path)
            # Add to list
            logits.append(logit)

        return behavior, numpy.array(logits)

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

                    print(f"Processing {sequence_path}")

                    # For every Frame in Sequence Folder
                    for frame in os.listdir(sequence_path):

                        if frame == "logits_temp":  # If iterated file is the features_temp folder, skip
                            continue

                        frame_path = os.path.join(sequence_path, frame)
                        # Add Frame to List
                        frame_list.append(frame_path)

                        save_path = sequence_path + "/logits_temp"

                        # If to get features; saves features to disk if True
                        if self.to_get_features:
                            # Preprocess then save to disk
                            self.hybrid_model.preprocess(
                                save_path, frame_path, frame)

                    self.dataset_data.append(
                        (behavior, save_path))  # Add to dataset_data

        print(f"Total number of processed sequences: {len(self.dataset_data)}")
