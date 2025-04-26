import random
import clip
import numpy
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import PIL
import re
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
_LSTM_INPUT_SIZE = 256  # Changed to match adapter output
_LSTM_HIDDEN_SIZE = 128
_LSTM_NUM_LAYERS = 2
_VIEW_EMBEDDING_DIM = 64
_CLIP_OUTPUT_DIM = 512

_BEHAVIOR_LABEL = {
    "a": 0,  # Safe Driving
    "b": 1,  # Texting
    "c": 2,  # Talking using Phone
    "d": 3,  # Drinking
    "e": 4,  # Head Down
    "f": 5,  # Look Behind
}


def natural_sort_key(s):
    """Convert string with numbers into tuple of strings and integers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


class HybridModel(nn.Module):
    """The class of the CLIP-LSTM hybrid used for distracted driving detection."""

    def __init__(self, use_precomputed: bool):
        """Initializes instance of CLIP-LSTM hybrid model"""
        # REQUIRED; Initializing parent class
        super().__init__()

        # Safer parameters if original ones cause issues
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.1,  # reduced from 0.2
                contrast=0.1,    # reduced from 0.2
                saturation=0.1,  # reduced from 0.2
                hue=0.05        # reduced from 0.1
            ),
            transforms.RandomAffine(
                degrees=(-3, 3),  # reduced from (-5, 5)
                translate=(0.05, 0.05),  # reduced from (0.1, 0.1)
                scale=(0.95, 1.05)  # reduced from (0.9, 1.1)
            ),
            transforms.RandomPerspective(
                distortion_scale=0.1, p=0.2),  # reduced values
            transforms.RandomGrayscale(p=0.05)  # reduced from 0.1
        ])

        self.use_precomputed = use_precomputed  # Use precomputed features or not
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

        # CLIP model outputs 512-dimensional features
        self.clip_model, self.preprocessor = clip.load(
            _MODEL, device=_DEVICE, jit=False)
        self.preprocessor = self.get_letterbox_preprocessor()

        print("Loading LSTM model...")

        # View embedding (2 views -> 64 dim)
        self.view_embedding = nn.Embedding(
            num_embeddings=2,  # front/side view
            embedding_dim=_VIEW_EMBEDDING_DIM,
            device=_DEVICE
        )

        # View attention (64 -> 512)
        self.view_attention = nn.Sequential(
            nn.Linear(_VIEW_EMBEDDING_DIM, _CLIP_OUTPUT_DIM),
            nn.Tanh(),
            nn.Linear(_CLIP_OUTPUT_DIM, _CLIP_OUTPUT_DIM),
            nn.Sigmoid()
        )

        # Adapter (512 + 64 -> 256)
        self.adapter = nn.Sequential(
            nn.Linear(_CLIP_OUTPUT_DIM +
                      _VIEW_EMBEDDING_DIM, _LSTM_INPUT_SIZE),
            nn.LayerNorm(_LSTM_INPUT_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # LSTM (256 -> 128)
        self.lstm = nn.LSTM(
            input_size=_LSTM_INPUT_SIZE,
            hidden_size=_LSTM_HIDDEN_SIZE,
            num_layers=_LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
            device=_DEVICE
        )

        # Temporal attention (128 -> 1)
        self.temporal_attention = nn.Sequential(
            nn.Linear(_LSTM_HIDDEN_SIZE, _LSTM_HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(_LSTM_HIDDEN_SIZE, 1)
        )

        self.temporal_dropout = nn.Dropout(0.2)

        # Final classification (128 -> num_classes)
        self.fc = nn.Linear(_LSTM_HIDDEN_SIZE, _NUM_OF_CLASSES, device=_DEVICE)

        print(f"Successfully Loaded! Using device: {_DEVICE}")

    def extract_features(self, image: Image.Image):
        """Process image through CLIP and return the 512-dim feature"""
        preprocessed = self.preprocessor(image).unsqueeze(0).to(_DEVICE)
        return self.clip_model.encode_image(preprocessed).squeeze(0)

    def set_clip_grad_status(self, stage):
        """
        Set which CLIP layers should have gradients enabled
        stage: 0 = all frozen, 1 = last few unfrozen, 2 = all unfrozen
        """
        # First freeze everything
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if stage >= 1:
            # Unfreeze last few transformer blocks
            layers_to_unfreeze = 2
            for i, block in enumerate(reversed(self.clip_model.visual.transformer.resblocks)):
                if i < layers_to_unfreeze:
                    for param in block.parameters():
                        param.requires_grad = True

        if stage >= 2:
            # Unfreeze all CLIP layers
            for param in self.clip_model.parameters():
                param.requires_grad = True

    def forward(self, tensor_sequence, view_type):
        # tensor_sequence shape: [batch_size, seq_len, 512]
        batch_size, seq_len, feat_dim = tensor_sequence.shape

        # View embedding: [batch_size, 64] -> [batch_size, seq_len, 64]
        view_emb = self.view_embedding(view_type.long())
        view_emb = view_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # View attention: [batch_size, seq_len, 512]
        view_attention = self.view_attention(view_emb)
        attended_features = tensor_sequence * view_attention

        # Combine features: [batch_size, seq_len, 576]
        combined = torch.cat([attended_features, view_emb], dim=2)

        # Adapter: [batch_size * seq_len, 256]
        flattened = combined.view(-1, _CLIP_OUTPUT_DIM + _VIEW_EMBEDDING_DIM)
        adapted = self.adapter(flattened)
        adapted_sequence = adapted.view(batch_size, seq_len, _LSTM_INPUT_SIZE)

        # LSTM: [batch_size, seq_len, 128]
        lstm_output, _ = self.lstm(adapted_sequence)

        # Temporal attention: [batch_size, seq_len, 1]
        attention_weights = self.temporal_attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum: [batch_size, 128]
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)

        # Final classification: [batch_size, num_classes]
        output = self.fc(attended_output)
        return output

    def preprocess(self, save_directory: str, frame_path: str, frame_name: str, view_type: int, sequence_id: str = None):
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
        if True:
            # If this is a new sequence, generate new augmentation parameters
            if sequence_id != self.current_sequence_id:
                self.current_sequence_id = sequence_id
                self._reset_sequence_params()

            # Apply consistent augmentation
            image = self._apply_sequence_augmentation(image)

        preprocessed = self.preprocessor(image).unsqueeze(
            0).to(_DEVICE)  # Open image, preprocess then save to device

        # print(f"Preprocessed: {preprocessed.shape}"); torch.Size([1, 3, 224, 224])

        # Edit dimension then convert to numpy
        preprocessed = preprocessed.cpu().numpy()
        save_name = os.path.join(
            save_directory, f"{frame_name.replace('.jpg', '')}_view{view_type}")
        numpy.save(save_name, preprocessed)  # Save preprocessed to disk

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
        behavior, preprocessed_path, view_type = self.dataset_data[index]
        preprocesseds = []
        for preprocessed in sorted(os.listdir(preprocessed_path), key=natural_sort_key):
            if not preprocessed.endswith(".npy"):
                continue

            # Open saved preprocessed image
            path = os.path.join(preprocessed_path, preprocessed)
            preprocessed = numpy.load(path)
            # Convert to tensor and add to list
            preprocessed = torch.tensor(preprocessed, device=_DEVICE)

            preprocesseds.append(self.hybrid_model.clip_model.encode_image(
                preprocessed).squeeze(0))  # Extract features then add to list

        return torch.tensor(behavior, device='cpu'), torch.stack(preprocesseds).cpu(), torch.tensor(view_type, device='cpu')

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
                    sequence_folders, key=natural_sort_key)  # Sort numerically

                # For every grouped sequence in current behavior folder
                for sequence_folder in sequence_folders:

                    # print(f"Sequence: {behavior}, {sequence_folder}")

                    # Folder of current behavior sequence
                    sequence_path = os.path.join(
                        behavior_path, sequence_folder)

                    view_type = 0 if "front" in sequence_path.lower() else 1

                    print(
                        f"Processing {sequence_path} (View type: {'front' if view_type == 0 else 'side'})")

                    # Get all frames and sort them alphanumerically
                    frames = os.listdir(sequence_path)
                    frames = sorted(frames, key=lambda x: x if x ==
                                    "features_temp" else x.lower())

                    sequence = []

                    # For every Frame in Sequence Folder
                    for frame in frames:

                        if frame == "features_temp":  # If iterated file is the features_temp folder, skip
                            continue

                        frame_path = os.path.join(sequence_path, frame)

                        save_path = sequence_path + "/features_temp"

                        # If to get features; saves features to disk if True
                        if self.to_get_features and self.hybrid_model.use_precomputed:
                            # Pass sequence_folder as sequence_id
                            self.hybrid_model.preprocess(
                                save_path, frame_path, frame, view_type, sequence_id=sequence_folder)
                        else:
                            sequence.append(frame_path)

                    if self.hybrid_model.use_precomputed:
                        self.dataset_data.append(
                            # Add to dataset_data
                            (behavior, save_path, view_type))
                    else:
                        self.dataset_data.append(
                            # Add to dataset_data
                            (behavior, sequence, view_type))

        print(f"Total number of processed sequences: {len(self.dataset_data)}")
        # Log distribution of views
        front_count = sum(1 for _, _, v in self.dataset_data if v == 0)
        side_count = sum(1 for _, _, v in self.dataset_data if v == 1)
        print(
            f"Front view sequences: {front_count}, Side view sequences: {side_count}")


if __name__ == "__main__":
    # Example usage
    hybrid_model = HybridModel()

    hybrid_model.visualize_letterbox("./CLIP.jpg")  # Path to your image
