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
from torch.nn.functional import cosine_similarity

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

        # Update classifier layers to match the saved model dimensions
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, texts):
        with torch.no_grad():
            # Get CLIP features and normalize them
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(texts)
            
            # Convert to float32
            image_features = image_features.float()
            text_features = text_features.float()
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Concatenate features to create input for classifier
            # [batch_size, 512] + [batch_size, 512] = [batch_size, 1024]
            combined_features = torch.cat([image_features, text_features], dim=1)
            
            # Pass through classifier
            output = self.classifier(combined_features)
            
        return output


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

# Load the trained weights and convert to float32
state_dict = torch.load('custom_clip_model.pth')
state_dict = {k: v.float() for k, v in state_dict.items()}
test_model.load_state_dict(state_dict)

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

# Add testing function
def test_model_cosine(model, test_loader):
    """Test the model using cosine similarity and classifier"""
    model.eval()
    all_predictions = []
    all_labels = []
    total = 0
    
    print("\nTesting model...")
    with torch.no_grad():
        for batch_images, batch_texts in tqdm(test_loader):
            # Ensure float32 type and move to device
            batch_images = batch_images.float().to(device)
            batch_size = batch_images.size(0)
            similarities_per_category = []
            
            # Get predictions for each category
            for prompt in DRIVING_CATEGORIES_PROMPTS:
                category_text = clip.tokenize([prompt] * batch_size).to(device)
                output = model(batch_images, category_text)
                similarities_per_category.append(output.cpu().numpy())
            
            # Convert to numpy array and get predicted category
            similarities_per_category = np.array(similarities_per_category).squeeze().T
            predicted_categories = np.argmax(similarities_per_category, axis=1)
            
            # Get true labels
            true_labels = [
                next(i for i, cat in enumerate(['c0', 'c1', 'c2', 'c3', 'c4', 'c5']) 
                     if cat in path) 
                for path in test_loader.dataset.image_paths[total:total + batch_size]
            ]
            
            # Store predictions and labels
            all_predictions.extend(predicted_categories)
            all_labels.extend(true_labels)
            total += batch_size
    
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    # Calculate per-category metrics
    categories = ['Safe Driving', 'Texting', 'Talking', 'Radio', 'Drinking', 'Reaching Behind']
    print("\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    print("\nPer-Category Performance:")
    for i, category in enumerate(categories):
        category_indices = [idx for idx, label in enumerate(all_labels) if label == i]
        if category_indices:
            category_correct = sum(1 for idx in category_indices if all_predictions[idx] == i)
            category_accuracy = category_correct / len(category_indices)
            print(f"{category}: {category_accuracy:.4f}")
            
            # Calculate confusion with other categories
            confusions = {}
            for j, other_cat in enumerate(categories):
                if j != i:
                    wrong_predictions = sum(1 for idx in category_indices if all_predictions[idx] == j)
                    if wrong_predictions > 0:
                        confusions[other_cat] = wrong_predictions / len(category_indices)
            
            if confusions:
                print("  Most confused with:")
                sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
                for confused_cat, rate in sorted_confusions[:2]:  # Show top 2 confusions
                    print(f"    - {confused_cat}: {rate:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'similarities': similarities_per_category
    }

# Modify main execution
if __name__ == "__main__":
    # Create test dataset and loader
    test_directory = "./testing_dataset"
    test_image_paths, test_text_descriptions = get_dataset_object(test_directory)
    
    test_dataset = ImageTextDataset(test_image_paths, test_text_descriptions)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test the model
    results = test_model_cosine(test_model, test_loader)
