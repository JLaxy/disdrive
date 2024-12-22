import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import clip
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define LSTM for Smoothing Predictions
class PredictionSmoothingLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(PredictionSmoothingLSTM, self).__init__()
        self.lstm = nn.LSTM(num_classes, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM outputs smoothed features
        out = self.fc(out)    # Map back to class logits
        return out

# Define the dataset class
class DrivingDataset(Dataset):
    def __init__(self, root_dir, sequence_length, preprocess, classes):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.preprocess = preprocess
        self.classes = classes
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for class_label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            video_frames = sorted(os.listdir(class_dir))
            
            # Group frames into sequences
            for i in range(0, len(video_frames) - self.sequence_length + 1, self.sequence_length):
                sequence = video_frames[i:i + self.sequence_length]
                self.data.append((sequence, class_label, class_dir))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label, class_dir = self.data[idx]
        frames = []
        for frame in sequence:
            image_path = os.path.join(class_dir, frame)
            image = read_image(image_path)
            image = self.preprocess(Image.fromarray(image.numpy().transpose(1, 2, 0)))
            frames.append(image)
        
        # Combine frames into a tensor
        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor, label

# Parameters
sequence_length = 10
num_classes = 5  # Example: Focused + 4 distracted behaviors
hidden_dim = 64
batch_size = 16

# Initialize dataset and data loader
dataset = DrivingDataset(
    root_dir="dataset",  # Update with the actual dataset path
    sequence_length=sequence_length,
    preprocess=preprocess,
    classes=["Focused", "Texting", "Eating", "Talking", "Reaching"]
)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the LSTM model
smoothing_model = PredictionSmoothingLSTM(num_classes, hidden_dim).to(device)

# Optimizer and loss function
optimizer = Adam(smoothing_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):  # Number of epochs
    smoothing_model.train()
    total_loss = 0

    for batch in data_loader:
        frames, labels = batch
        frames, labels = frames.to(device), labels.to(device)
        
        # Extract features with CLIP
        batch_logits = []
        for sequence in frames:  # Iterate over sequences
            logits = []
            for frame in sequence:  # Iterate over frames
                with torch.no_grad():
                    logit = clip_model.encode_image(frame.unsqueeze(0).to(device))
                    logits.append(logit.squeeze(0))
            batch_logits.append(torch.stack(logits, dim=0))  # Sequence logits
        
        batch_logits = torch.stack(batch_logits, dim=0)  # [batch_size, sequence_length, feature_dim]

        # Train the LSTM
        optimizer.zero_grad()
        outputs = smoothing_model(batch_logits)  # Shape: [batch_size, sequence_length, num_classes]
        loss = criterion(outputs[:, -1, :], labels)  # Compare last time step predictions to ground truth
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{10}, Loss: {total_loss / len(data_loader)}")

# Save the model
torch.save(smoothing_model.state_dict(), "smoothing_model.pth")

# Evaluation (Example)
smoothing_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in data_loader:
        frames, labels = batch
        frames, labels = frames.to(device), labels.to(device)
        
        # Extract features with CLIP
        batch_logits = []
        for sequence in frames:
            logits = []
            for frame in sequence:
                logit = clip_model.encode_image(frame.unsqueeze(0).to(device))
                logits.append(logit.squeeze(0))
            batch_logits.append(torch.stack(logits, dim=0))
        
        batch_logits = torch.stack(batch_logits, dim=0)
        outputs = smoothing_model(batch_logits)
        preds = torch.argmax(outputs[:, -1, :], dim=-1)  # Predictions from last time step
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Focused", "Texting", "Eating", "Talking", "Reaching"]))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
