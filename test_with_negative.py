from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import clip
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns

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

# Trainable CLIP Classifier model The new layer is added to the end of the model


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


# Create a new instance of your model
test_model = DisDriveClassifier(model).to(device)
# Load the trained weights
test_model.load_state_dict(torch.load('best_model.pth'))


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


# Trainable CLIP Classifier model The new layer is added to the end of the model
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

# Dataset for image-text pairs; a container/datastructure for the image paths and text descriptions pairs


class ImageTextDataset(Dataset):
    def __init__(self, image_paths, text_descriptions, preprocess=preprocess):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.preprocess = preprocess

    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)

        # Get corresponding text
        text = self.text_descriptions[idx]
        text = clip.tokenize(text).squeeze(0)

        return image, text

    def __len__(self):
        return len(self.image_paths)


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


def evaluate_model(model, test_loader):
    """Test the accuracy of the imported model against all categories"""
    model.eval()
    print("\nTesting model...")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, texts in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            texts = texts.to(device)

            # Get model predictions for each category
            predictions_per_category = []
            for category_prompt in tokenized_driving_categories_prompts:
                outputs = model(images, category_prompt.repeat(images.size(0), 1))
                predictions = torch.sigmoid(outputs).cpu().numpy()
                predictions_per_category.append(predictions)
            
            # Stack predictions for all categories
            predictions_per_category = np.stack(predictions_per_category, axis=1)
            # Get the category with highest probability
            predictions = np.argmax(predictions_per_category, axis=1)

            # Get true labels by finding the most similar prompt
            labels = []
            for text in texts:
                text_features = model.clip_model.encode_text(text.unsqueeze(0))
                similarities = []
                for prompt in tokenized_driving_categories_prompts:
                    prompt_features = model.clip_model.encode_text(prompt.unsqueeze(0))
                    similarity = torch.cosine_similarity(text_features, prompt_features)
                    similarities.append(similarity.item())
                label = np.argmax(similarities)
                labels.append(label)

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics for each category
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, 
        labels=range(len(DRIVING_CATEGORIES_PROMPTS)))
    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Print results
    print("\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    
    # Print metrics for each category
    for i, category in enumerate(DRIVING_CATEGORIES_PROMPTS):
        print(f"Category {i} ({category.split('a photo of')[1].strip()}):")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1 Score: {f1[i]:.4f}\n")

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"C{i}" for i in range(len(DRIVING_CATEGORIES_PROMPTS))],
                yticklabels=[f"C{i}" for i in range(len(DRIVING_CATEGORIES_PROMPTS))])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    dataset_directory = "./testing_dataset"
    # Get dataset info
    image_paths, text_descriptions = get_dataset_object(dataset_directory)

    # Create dataset and dataloader
    dataset = ImageTextDataset(image_paths, text_descriptions)
    test_loader = DataLoader(dataset, batch_size=32,
                             shuffle=True, pin_memory=True)

    # Test the model
    evaluate_model(test_model, test_loader)
