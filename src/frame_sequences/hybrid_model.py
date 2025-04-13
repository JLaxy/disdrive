import random
import clip
import numpy
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import PIL
from torchvision import transforms
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt

"""Hybrid Model Settings"""
_MODEL = "ViT-B/16"  # 224x224
# Automatically changes to GPU if available
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Number of Distracted Driving Behaviors"""
_NUM_OF_CLASSES = 6

"""LSTM Parameters"""
_LSTM_INPUT_SIZE = 256
_LSTM_HIDDEN_SIZE = 256
_LSTM_NUM_LAYERS = 2

_BEHAVIOR_LABEL = {
    "a": 0,  # Safe Driving
    "b": 1,  # Texting
    "c": 2,  # Talking using Phone
    "d": 3,  # Drinking
    "e": 4,  # Head Down
    "f": 5,  # Look Behind
}


class HybridModel(nn.Module):
    """The class of the CLIP-LSTM hybrid used for distracted driving detection."""

    def __init__(self):
        """Initializes instance of CLIP-LSTM hybrid model"""
        # REQUIRED; Initializing parent class
        super().__init__()

        # # Add to model initialization
        # self.augmentation = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.3),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2)
        # ])

        # # Safer parameters if original ones cause issues
        # self.augmentation = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.3),
        #     transforms.ColorJitter(
        #         brightness=0.1,  # reduced from 0.2
        #         contrast=0.1,    # reduced from 0.2
        #         saturation=0.1,  # reduced from 0.2
        #         hue=0.05        # reduced from 0.1
        #     ),
        #     transforms.RandomAffine(
        #         degrees=(-3, 3),  # reduced from (-5, 5)
        #         translate=(0.05, 0.05),  # reduced from (0.1, 0.1)
        #         scale=(0.95, 1.05)  # reduced from (0.9, 1.1)
        #     ),
        #     transforms.RandomPerspective(distortion_scale=0.1, p=0.2),  # reduced values
        #     transforms.RandomGrayscale(p=0.05)  # reduced from 0.1
        # ])

        self.current_sequence_id = None
        self.sequence_params = {
            'flip': False,
            'brightness': 0.0,
            'contrast': 0.0,
            'saturation': 0.0,
            'hue': 0.0,
            'angle': 0.0,
            'translate': (0.0, 0.0),
            'scale': 1.0
        }

        print("Loading CLIP model...")

        # Loading CLIP model
        self.clip_model, self.preprocessor = clip.load(
            _MODEL, device=_DEVICE, jit=False)

        self.preprocessor = self.get_letterbox_preprocessor()

        print("Loading LSTM model...")

        # Adapter for CLIP model
        self.adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.temporal_dropout = nn.Dropout(0.2)

        # Initalizing LSTM Neural Network
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            input_size=_LSTM_INPUT_SIZE,
            hidden_size=_LSTM_HIDDEN_SIZE,
            num_layers=_LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
            device=_DEVICE
        )

        self.fc = nn.Linear(_LSTM_HIDDEN_SIZE, _NUM_OF_CLASSES, device=_DEVICE)

        print(f"Successfully Loaded! Using device: {_DEVICE}")

    def forward(self, tensor_sequence):
        batch_size, seq_len, feat_dim = tensor_sequence.shape

        flattened = tensor_sequence.view(-1, feat_dim)
        adapted = self.adapter(flattened)
        adapted_sequence = adapted.view(batch_size, seq_len, -1)

        # Dropout before LSTM (temporal regularization)
        adapted_sequence = self.temporal_dropout(adapted_sequence)

        lstm_output, _ = self.lstm(adapted_sequence)
        mean_pooled = lstm_output.mean(dim=1)  # average over time
        output = self.fc(mean_pooled)

        return output

    def preprocess(self, save_directory: str, frame_path: str, frame_name: str, sequence_id: str = None):
        """
        Preprocess image then save to disk
        Args:
            save_directory: Directory to save processed features
            frame_path: Path to the frame image
            frame_name: Name of the frame file
            sequence_id: Identifier for the sequence this frame belongs to
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        image = Image.open(frame_path)

        # Apply augmentation before CLIP preprocessing
        if False:
            # If this is a new sequence, generate new augmentation parameters
            if sequence_id != self.current_sequence_id:
                self.current_sequence_id = sequence_id
                self._reset_sequence_params()

            # Apply consistent augmentation
            image = self._apply_sequence_augmentation(image)

        preprocessed = self.preprocessor(image).unsqueeze(
            0).to(_DEVICE)  # Open image, preprocess then save to device

        # preprocessed = self.preprocessor(Image.open(frame_path)).unsqueeze(
        #     0).to(_DEVICE)  # Open image, preprocess then save to device

        with torch.no_grad():
            features = self.clip_model.encode_image(
                preprocessed)  # Extract features

        # Edit dimension then convert to numpy
        features = features.squeeze(0).cpu().numpy()
        numpy.save(os.path.join(save_directory, frame_name.replace(".jpg", "")),
                   features)  # Save feature to disk

    # Letterbox helper
    def letterbox_image(self, image: Image.Image, size=(224, 224), fill_color=(0, 0, 0)):
        image = image.convert("RGB")  # Ensure RGB
        # Resize with aspect ratio preserved
        image.thumbnail(size, Image.BICUBIC)

        # Compute padding
        delta_w = size[0] - image.size[0]
        delta_h = size[1] - image.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w -
                   delta_w // 2, delta_h - delta_h // 2)

        # Add padding
        return ImageOps.expand(image, padding, fill=fill_color)

    # Compose the full preprocessor (like CLIP but with letterbox)
    def get_letterbox_preprocessor(self):
        return Compose([
            lambda img: self.letterbox_image(img, (224, 224)),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711))
        ])

    def visualize_letterbox(self, image_path):
        # Load original image
        original = Image.open(image_path).convert("RGB")

        # Apply letterbox padding
        padded = self.letterbox_image(original, size=(224, 224))

        # Show side by side
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(original)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(padded)
        axs[1].set_title("Letterbox Padded (224x224)")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    def _reset_sequence_params(self):
        """Reset augmentation parameters for new sequence with more robust values"""
        self.sequence_params = {
            'flip': random.random() < 0.5,  # Increased from 0.3
            'brightness': random.uniform(-0.2, 0.2),  # Increased from ±0.1
            'contrast': random.uniform(-0.2, 0.2),    # Increased from ±0.1
            'saturation': random.uniform(-0.2, 0.2),  # Increased from ±0.1
            'hue': random.uniform(-0.1, 0.1),        # Increased from ±0.05
            'angle': random.uniform(-5, 5),          # Increased from ±3
            'translate': (
                random.uniform(-0.1, 0.1),           # Increased from ±0.05
                random.uniform(-0.1, 0.1)
            ),
            'scale': random.uniform(0.9, 1.1)        # Increased from 0.95-1.05
        }

    def _apply_sequence_augmentation(self, image):
        """Apply consistent augmentation to frame"""
        if self.sequence_params['flip']:
            image = transforms.functional.hflip(image)

        image = transforms.functional.adjust_brightness(
            image, 1 + self.sequence_params['brightness'])
        image = transforms.functional.adjust_contrast(
            image, 1 + self.sequence_params['contrast'])
        image = transforms.functional.adjust_saturation(
            image, 1 + self.sequence_params['saturation'])
        image = transforms.functional.adjust_hue(
            image, self.sequence_params['hue'])

        image = transforms.functional.affine(
            image,
            angle=self.sequence_params['angle'],
            translate=self.sequence_params['translate'],
            scale=self.sequence_params['scale'],
            shear=0
        )

        return image


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
        self.dataset_data = []  # [(behavior, [IMAGE_SEQUENCE])]

        # Process Dataset
        self.__process_dataset()

    def __getitem__(self, index):
        behavior, feature_path = self.dataset_data[index]
        features = []

        for feature_file in sorted(os.listdir(feature_path)):
            if not feature_file.endswith(".npy"):
                continue
            path = os.path.join(feature_path, feature_file)
            feature = numpy.load(path)
            features.append(feature)

        return torch.tensor(behavior), torch.tensor(features, dtype=torch.float32)

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

                sequence_folders = os.listdir(behavior_path)
                sequence_folders = sorted(
                    sequence_folders, key=lambda x: int(x))  # Sort numerically

                # For every grouped sequence in current behavior folder
                for sequence_folder in sequence_folders:

                    # print(f"Sequence: {behavior}, {sequence_folder}")

                    # Folder of current behavior sequence
                    sequence_path = os.path.join(
                        behavior_path, sequence_folder)

                    frame_list = []  # List of frames in a sequence of behavior

                    print(f"Processing {sequence_path}")

                    # Get all frames and sort them alphanumerically
                    frames = os.listdir(sequence_path)
                    frames = sorted(frames, key=lambda x: x if x ==
                                    "features_temp" else x.lower())

                    # For every Frame in Sequence Folder
                    for frame in frames:

                        if frame == "features_temp":  # If iterated file is the features_temp folder, skip
                            continue

                        frame_path = os.path.join(sequence_path, frame)
                        # Add Frame to List
                        frame_list.append(frame_path)

                        save_path = sequence_path + "/features_temp"

                        # If to get features; saves features to disk if True
                        if self.to_get_features:
                            # Pass sequence_folder as sequence_id
                            self.hybrid_model.preprocess(
                                save_path, frame_path, frame, sequence_id=sequence_folder)

                    self.dataset_data.append(
                        (behavior, save_path))  # Add to dataset_data

        print(f"Total number of processed sequences: {len(self.dataset_data)}")


if __name__ == "__main__":
    # Example usage
    hybrid_model = HybridModel()

    hybrid_model.visualize_letterbox("./CLIP.jpg")  # Path to your image
