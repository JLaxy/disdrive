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


class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess=preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __getitem__(self, idx):
        # Load and process image only
        image = Image.open(self.image_paths[idx])
        image = self.preprocess(image)
        return image

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

# Example usage:
if __name__ == "__main__":
    # Create test dataset and loader
    test_directory = "./testing_dataset"
    test_image_paths, _ = get_dataset_object(test_directory)  # We don't need text_descriptions

    test_dataset = ImageDataset(test_image_paths)  # Simplified dataset class
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def test_imported_model(model, test_loader):
        """Test the accuracy of the imported model"""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        print("\nTesting model...")
        with torch.no_grad():
            for batch_images in tqdm(test_loader):
                # Move data to device
                batch_images = batch_images.to(device)
                batch_size = batch_images.size(0)
                
                # Compare each image against all category prompts
                category_outputs = []
                for prompt in DRIVING_CATEGORIES_PROMPTS:
                    category_text = clip.tokenize([prompt] * batch_size).to(device)
                    output = model(batch_images, category_text)
                    category_outputs.append(output)
                
                # Stack outputs and get predicted category
                category_outputs = torch.stack(category_outputs, dim=1).squeeze(-1)
                predictions = torch.argmax(category_outputs, dim=1)
                
                # Get true labels (0 for c0, 1 for c1, etc.)
                true_labels = torch.tensor([
                    next(i for i, cat in enumerate(['c0', 'c1', 'c2', 'c3', 'c4', 'c5']) 
                         if cat in path)
                    for path in test_loader.dataset.image_paths[total:total + batch_size]
                ]).to(device)
                
                # Update counters
                correct += (predictions == true_labels).sum().item()
                total += true_labels.size(0)
                
                # Store predictions and labels for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
            
            # Calculate overall accuracy
            accuracy = correct / total
            
            # Calculate per-category metrics
            categories = ['Safe Driving', 'Texting', 'Talking', 'Radio', 'Drinking', 'Reaching Behind']
            
            print("\nTest Results:")
            print(f"Total Images Tested: {total}")
            print(f"Correct Predictions: {correct}")
            print(f"Overall Accuracy: {accuracy:.4f}")
            
            print("\nPer-Category Performance:")
            for i, category in enumerate(categories):
                category_indices = [idx for idx, label in enumerate(all_labels) if label == i]
                if category_indices:
                    category_correct = sum(1 for idx in category_indices if all_predictions[idx] == i)
                    category_accuracy = category_correct / len(category_indices)
                    print(f"\n{category}:")
                    print(f"Accuracy: {category_accuracy:.4f}")
                    
                    # Show confusion with other categories
                    confusions = {}
                    for j, other_cat in enumerate(categories):
                        if j != i:
                            wrong_predictions = sum(1 for idx in category_indices if all_predictions[idx] == j)
                            if wrong_predictions > 0:
                                confusion_rate = wrong_predictions / len(category_indices)
                                confusions[other_cat] = confusion_rate
                    
                    if confusions:
                        print("Most confused with:")
                        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
                        for confused_cat, rate in sorted_confusions[:2]:
                            print(f"  - {confused_cat}: {rate:.4f}")
            
            return {
                'accuracy': accuracy,
                'predictions': all_predictions,
                'true_labels': all_labels,
                'per_category_metrics': {
                    cat: {
                        'accuracy': sum(1 for idx in [i for i, l in enumerate(all_labels) if l == j]
                                      if all_predictions[idx] == j) / len([i for i, l in enumerate(all_labels) if l == j])
                        if len([i for i, l in enumerate(all_labels) if l == j]) > 0 else 0
                    } for j, cat in enumerate(categories)
                }
            }
    
    # Test the imported model
    results = test_imported_model(test_model, test_loader)
