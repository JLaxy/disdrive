import torch
import clip
import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Load the original CLIP model first
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# Prompts for distracted driving behaviors
DRIVING_CATEGORIES_PROMPTS = [
    "a photo of a person driving safely with both hands on the steering wheel",  # c0
    "a photo of a person texting using their mobile phone while driving",        # c1
    "a photo of a person talking on the phone while driving",                    # c2
    "a photo of a person operating the car radio while driving",                 # c3
    "a photo of a person drinking while driving",                                # c4
    "a photo of a person reaching for an object behind them while driving",      # c5
]


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


# Create a new instance of your model
test_model = DisDriveClassifier(clip_model).to(device)

# Load the trained weights
test_model.load_state_dict(torch.load('custom_clip_model.pth'))

# Set model to evaluation mode
test_model.eval()


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


def test_model_on_dataset(model, test_loader):
    """Function to test the model on a dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    categories = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']

    # Create progress bar
    progress_bar = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for batch_images, batch_texts in progress_bar:
            batch_images = batch_images.to(device)

            # Test against all categories for each image
            batch_size = batch_images.size(0)
            batch_predictions = []

            # Get predictions for each category
            for prompt in DRIVING_CATEGORIES_PROMPTS:
                # Prepare text input
                category_text = clip.tokenize([prompt] * batch_size).to(device)

                # Get predictions
                similarities = model(batch_images, category_text)
                predictions = torch.sigmoid(similarities)
                batch_predictions.append(predictions.cpu().numpy())

            # Convert to numpy array and get max prediction
            batch_predictions = np.array(batch_predictions).squeeze().T
            predicted_categories = np.argmax(batch_predictions, axis=1)

            # Get true labels from file paths
            batch_paths = test_loader.dataset.image_paths[len(
                all_predictions):len(all_predictions) + batch_size]
            true_categories = [next(i for i, cat in enumerate(
                categories) if cat in path) for path in batch_paths]

            # Store predictions and labels
            all_predictions.extend(predicted_categories)
            all_labels.extend(true_categories)

            # Calculate running accuracy
            current_acc = sum([1 for p, l in zip(all_predictions, all_labels)
                               if p == l]) / len(all_predictions)

            # Update progress bar
            progress_bar.set_postfix({
                'accuracy': f'{current_acc:.4f}'
            })

    # Calculate final metrics
    accuracy = sum([1 for p, l in zip(all_predictions, all_labels)
                   if p == l]) / len(all_labels)

    # Calculate per-category metrics
    category_metrics = {}
    for cat_idx, category in enumerate(categories):
        cat_indices = [i for i, l in enumerate(all_labels) if l == cat_idx]
        if cat_indices:
            # Calculate metrics for this category
            cat_true = [1 if l == cat_idx else 0 for l in all_labels]
            cat_pred = [1 if p == cat_idx else 0 for p in all_predictions]

            precision, recall, f1, _ = precision_recall_fscore_support(
                cat_true, cat_pred, average='binary'
            )

            category_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': sum([1 for i in cat_indices if all_predictions[i] == cat_idx]) / len(cat_indices)
            }

    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )

    print("\nOverall Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nPer-Category Metrics:")
    for cat, metrics in category_metrics.items():
        print(f"\n{cat}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    return {
        'overall_accuracy': accuracy,
        'category_metrics': category_metrics,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


# Example usage:
if __name__ == "__main__":
    # Create test dataset and loader
    test_directory = "./training_dataset"
    test_image_paths, test_text_descriptions = get_dataset_object(
        test_directory)

    test_dataset = ImageTextDataset(test_image_paths, test_text_descriptions)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test the model
    results = test_model_on_dataset(test_model, test_loader)

    # Print results
    print("\nTesting Results:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print("\nPer-Category Accuracies:")
    for cat, acc in results['category_metrics'].items():
        print(f"{cat}: {acc['accuracy']:.4f}")
