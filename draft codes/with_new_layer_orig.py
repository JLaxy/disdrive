from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import clip
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Selecting which device to use; CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# Prompts for distracted driving behaviors
DRIVING_CATEGORIES_PROMPTS = [
    "a photo of a person driving safely with both hands on the steering wheel",  # c0
    "a photo of a person texting using their mobile phone while driving",        # c1
    "a photo of a person talking on the phone while driving",                    # c2
    "a photo of a person operating the car radio while driving",                 # c3
    "a photo of a person drinking while driving",                                # c4
    "a photo of a person reaching for an object behind them while driving",      # c5
]
# Tokenize the prompts
tokenized_driving_categories_prompts = clip.tokenize(
    DRIVING_CATEGORIES_PROMPTS).to(device)


def get_image_embeddings(image):
    """Returns the image embeddings for a single image; resize the image to 336x336 before processing."""
    # Preprocess the image
    processed_image = preprocess(image).unsqueeze(0).to(device)
    # Get the image embeddings
    with torch.no_grad():
        image_embeddings = model.encode_image(processed_image)
    return image_embeddings


def get_text_embeddings(text: str):
    """Returns the text embeddings for a single text."""
    # Process the text
    processed_text = clip.tokenize(text).to(device)
    # Get the text embeddings
    with torch.no_grad():
        text_embeddings = model.encode_text(processed_text)
    return text_embeddings


def get_distracted_driving_behavior_probabilities(image):
    """Returns its probablities for each distracted driving behavior."""
    # Open image then automatically resize it to 336x336
    image = preprocess(image).unsqueeze(0).to(device)

    # Calculate probabilities
    with torch.no_grad():
        logits_per_image, logits_per_text = model(
            image, tokenized_driving_categories_prompts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        for label, prob in zip(DRIVING_CATEGORIES_PROMPTS, np.ravel(probs)):
            print(f"{label}: {prob:.2%}")

        return np.ravel(probs)


# print(get_distracted_driving_behavior_probabilities(
#     Image.open("images/76385.jpg")))


# Modified CLIP Classifier model
class DisDriveClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Initialize new trainable layer with the same dtype as CLIP
        self.classifier = nn.Sequential(
            # Convert to float32
            nn.Linear(512 * 2, 512).to(dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256).to(dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(256, 1).to(dtype=torch.float32)
        )

    def forward(self, images, texts):
        with torch.no_grad():
            # Get CLIP features and convert to float32
            image_features = self.clip_model.encode_image(
                images).float()  # Convert to float32
            text_features = self.clip_model.encode_text(
                texts).float()    # Convert to float32

        # Combine features
        combined = torch.cat((image_features, text_features), dim=1)

        # Get similarity score
        similarity = self.classifier(combined)
        return similarity

# Dataset for image-text pairs


class ImageTextDataset(Dataset):
    def __init__(self, image_paths, text_descriptions, preprocess=preprocess):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.preprocess = preprocess

    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.image_paths[idx])
        image = self.preprocess(image)

        # Get corresponding text
        text = self.text_descriptions[idx]
        text = clip.tokenize(text).squeeze(0)

        return image, text

    def __len__(self):
        return len(self.image_paths)


# Training function
def train_model(model, train_loader, num_epochs=3):
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Calculate total steps for entire training
    total_steps = num_epochs * len(train_loader)
    print(f"Starting training: {num_epochs} epochs, {
          len(train_loader)} batches per epoch")
    print(f"Total training steps: {total_steps}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Create progress bar for this epoch
        progress_bar = tqdm(train_loader,
                            desc=f'Epoch {epoch+1}/{num_epochs}',
                            leave=True)

        for batch_idx, (batch_images, batch_texts) in enumerate(progress_bar):
            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)

            # Forward pass
            similarity = model(batch_images, batch_texts)

            # Assuming matching pairs are positive examples (label=1)
            labels = torch.ones(similarity.shape).to(device)

            # Calculate loss
            loss = criterion(similarity, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss and progress bar
            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)

            # Update progress bar description with loss
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'batch': f'{batch_idx+1}/{len(train_loader)}'
            })

        # Print epoch summary
        epoch_loss = total_loss / len(train_loader)
        print(
            f"\nEpoch {epoch+1}/{num_epochs} completed - Average Loss: {epoch_loss:.4f}")

        # Optional: Add a blank line between epochs
        if epoch < num_epochs - 1:
            print()

    print("\nTraining completed!")


def get_dataset_object(dataset_directory):
    """Retrieves information about the dataset items."""
    # Containers for image paths and text descriptions
    image_paths = []
    text_descriptions = []

    # Descriptions for each category
    category_descriptions = {
        "c0": DRIVING_CATEGORIES_PROMPTS[0],
        "c1": DRIVING_CATEGORIES_PROMPTS[1],
        "c1.1": DRIVING_CATEGORIES_PROMPTS[1],
        "c2": DRIVING_CATEGORIES_PROMPTS[2],
        "c2.1": DRIVING_CATEGORIES_PROMPTS[2],
        "c3": DRIVING_CATEGORIES_PROMPTS[3],
        "c4": DRIVING_CATEGORIES_PROMPTS[4],
        "c5": DRIVING_CATEGORIES_PROMPTS[5],
    }

    # For each subfolder in the dataset directory
    for category in os.listdir(dataset_directory):
        # Get category path
        category_path = os.path.join(dataset_directory, category)

        # Makes sure it is a directory
        if os.path.isdir(category_path):
            description = category_descriptions.get(category)
            # For each image in the category folder
            for image in os.listdir(category_path):
                # Get path
                img_path = os.path.join(category_path, image)
                # Add to list of image paths
                image_paths.append(img_path)
                # Add description to list of text descriptions
                text_descriptions.append(description)

    return image_paths, text_descriptions


if __name__ == "__main__":
    # Create custom model
    custom_model = DisDriveClassifier(model).to(device)

    dataset_directory = "./training_dataset"
    # Get dataset info
    image_paths, text_descriptions = get_dataset_object(dataset_directory)

    # Create dataset and dataloader
    dataset = ImageTextDataset(image_paths, text_descriptions)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    train_model(custom_model, train_loader)

    # Save the trained model
    torch.save(custom_model.state_dict(), 'custom_clip_model.pth')
