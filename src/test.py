import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
from torch.utils.data import DataLoader, Dataset, random_split
import clip
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm  # For progress bars

# Model definition
class DistractedDriverDetector(nn.Module):
    def __init__(self, num_classes=10):
        super(DistractedDriverDetector, self).__init__()
        
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Define text descriptions for each class
        self.class_descriptions = [
            "a person driving safely and attentively",              # c0
            "a driver texting with right hand while driving",       # c1
            "a driver talking on phone with right hand",            # c2
            "a driver texting with left hand while driving",        # c3
            "a driver talking on phone with left hand",             # c4
            "a driver operating the radio while driving",           # c5
            "a driver drinking while driving",                      # c6
            "a driver reaching behind while driving",               # c7
            "a driver doing hair and makeup while driving",         # c8
            "a driver talking to passenger while driving"           # c9
        ]
        
        # Encode text descriptions using CLIP
        text_tokens = clip.tokenize(self.class_descriptions).to(device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # LSTM parameters
        self.hidden_size = 512
        self.num_layers = 2
        
        # LSTM layer
        self.lstm = LSTM(
            input_size=512,  # CLIP feature dimension
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + 512, 256),  # Added text feature dimension
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, sequence_length = x.size(0), x.size(1)
        
        # Reshape for CLIP processing
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Extract CLIP image features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # Calculate similarity with text descriptions
        similarity = image_features @ self.text_features.t()
            
        # Reshape for LSTM
        image_features = image_features.view(batch_size, sequence_length, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(image_features)
        
        # Take final hidden state
        final_hidden = lstm_out[:, -1, :]
        
        # Concatenate LSTM output with text similarity features
        combined_features = torch.cat([final_hidden, similarity.mean(dim=0)], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output

    def update_class_descriptions(self, new_descriptions):
        """
        Update the text descriptions for classes
        Args:
            new_descriptions: list of strings, one for each class
        """
        if len(new_descriptions) != len(self.class_descriptions):
            raise ValueError("Number of new descriptions must match number of classes")
            
        self.class_descriptions = new_descriptions
        device = next(self.parameters()).device
        
        # Re-encode text descriptions
        text_tokens = clip.tokenize(self.class_descriptions).to(device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

# Dataset class for loading images
class AUCDriverDataset(Dataset):
    def __init__(self, root_dir, sequence_length=32, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory
            sequence_length (int): Number of frames to use per sequence
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match CLIP input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Get all classes (subfolders within 'train')
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Create list of (image_sequence, label) pairs
        self.sequences = []
        
        # Load sequences from the 'train' folder
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            image_files = sorted(glob.glob(os.path.join(class_path, '*.jpg')))
            
            # Group images into sequences
            for i in range(0, len(image_files) - sequence_length + 1, sequence_length):
                sequence = image_files[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    self.sequences.append((sequence, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_sequence, label = self.sequences[idx]
        frames = []
        
        # Load and transform each image in the sequence
        for img_path in image_sequence:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        # Stack frames into a single tensor
        frames = torch.stack(frames)
        return frames, label

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Evaluation function
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')
    return accuracy

# Main script
if __name__ == "__main__":
    # Set your dataset path
    dataset_path = "D:\\zek\\4th yr comsci\\CS 307\\data\\auc\\v2_cam1_cam2_split_by_driver\\Camera 1\\train"

    print("Dataset path:", dataset_path)
    print("Does the path exist?", os.path.exists(dataset_path))  # Check if the path exists

    # List the contents of the directory
    if os.path.exists(dataset_path):
        print("Directory contents:", os.listdir(dataset_path))
        
    # Create dataset for 'train' folder
    dataset = AUCDriverDataset(
        root_dir=dataset_path,
        sequence_length=32,  # Adjust based on your needs/memory constraints
    )
    
    # Debugging: Check if sequences were created
    print(f"Total number of sequences in the dataset: {len(dataset)}")

    # Split into train and test sets (though your data is already split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Initialize model, loss, optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DistractedDriverDetector(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate on the dataset
    print("Training model on dataset...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)
    
    print("Evaluating on test data...")
    evaluate_model(model, test_loader, device=device)