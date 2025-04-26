import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class DistractedDriverDataset(Dataset):
    def __init__(self, image_dir, labels=None, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images
            labels (dict): Dictionary mapping image filenames to labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(
            image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.labels = labels

        # CLIP standard preprocessing if no custom transform is provided
        if transform is None:
            self.transform = transforms.Compose([
                # Resize the shorter side to 256 pixels
                transforms.Resize(256),
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        image = self.transform(image)

        # Return label if available
        if self.labels:
            label = self.labels.get(img_name, 0)  # Default to 0 if not found
            return image, label
        else:
            return image

# Alternative preprocessing approaches


def get_preserve_aspect_ratio_transform():
    """
    Creates a transform that preserves aspect ratio by resizing then padding.
    This is useful when aspect ratio information is important for the task.
    """
    return transforms.Compose([
        # Fix: Use only size parameter or set max_size strictly greater than size
        transforms.Resize(224),  # Resize the smaller edge to 224 pixels
        transforms.Lambda(lambda img: pad_to_square(
            img, 224)),  # Pad to make it square
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])


def pad_to_square(img, target_size):
    """Pad image to square with black padding"""
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), (0, 0, 0))
        result.paste(img, (0, (width - height) // 2))
        return result.resize((target_size, target_size))
    else:
        result = Image.new(img.mode, (height, height), (0, 0, 0))
        result.paste(img, ((height - width) // 2, 0))
        return result.resize((target_size, target_size))


def get_augmentation_transform():
    """
    Creates a transform with data augmentation that might be useful 
    for training to improve model robustness.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

# Example usage


def load_data(data_dir, batch_size=32, preserve_aspect_ratio=True, use_augmentation=False):
    """
    Load and prepare the data with appropriate transforms

    Args:
        data_dir: Directory containing image data
        batch_size: Batch size for the data loader
        preserve_aspect_ratio: Whether to use aspect ratio preserving transforms
        use_augmentation: Whether to use data augmentation (for training)
    """
    if preserve_aspect_ratio:
        transform = get_preserve_aspect_ratio_transform()
    elif use_augmentation:
        transform = get_augmentation_transform()
    else:
        # Standard CLIP preprocessing (aspect ratio not preserved)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    dataset = DistractedDriverDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# Example to check if preprocessing is working correctly


def visualize_preprocessing(image_path):
    """Visualize the effect of preprocessing on an image"""
    import matplotlib.pyplot as plt

    original_img = Image.open(image_path).convert('RGB')

    # Standard CLIP transform
    clip_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Aspect ratio preserving transform
    aspect_transform = transforms.Compose([
        transforms.Resize(224),  # Removed max_size parameter
        transforms.Lambda(lambda img: pad_to_square(img, 224)),
        transforms.ToTensor(),
    ])

    clip_img = clip_transform(original_img)
    aspect_img = aspect_transform(original_img)

    # Convert to numpy for visualization
    clip_img = clip_img.permute(1, 2, 0).numpy()
    aspect_img = aspect_img.permute(1, 2, 0).numpy()

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_img)
    ax[0].set_title('Original Photo')
    ax[1].imshow(clip_img)
    ax[1].set_title('Standard CLIP Cropping')
    ax[2].imshow(aspect_img)
    ax[2].set_title('Letterbox Croppping')
    plt.show()


if __name__ == "__main__":
    # Visualize preprocessing
    visualize_preprocessing("./CLIP.jpg")
