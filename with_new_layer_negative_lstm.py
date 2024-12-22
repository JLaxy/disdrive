import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import clip
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# Distracted driving behavior prompts
DRIVING_CATEGORIES_PROMPTS = [
    "a photo of a person driving safely with both hands on the steering wheel",  # c0
    "a photo of a person texting using their mobile phone while driving",        # c1
    "a photo of a person talking on the phone while driving",                    # c2
    "a photo of a person operating the car radio while driving",                 # c3
    "a photo of a person drinking while driving",                                # c4
    "a photo of a person reaching for an object behind them while driving",      # c5
]
tokenized_driving_categories_prompts = clip.tokenize(DRIVING_CATEGORIES_PROMPTS).to(device)

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        if lstm_output.dim() == 2:
            lstm_output = lstm_output.unsqueeze(1)
        
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_output = torch.sum(weights * lstm_output, dim=1)
        return weighted_output

class MultiModalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        image_feature_dim = 512  # CLIP image feature size
        text_feature_dim = 512   # CLIP text feature size
        lstm_feature_dim = 256   # LSTM hidden size

        total_input_dim = image_feature_dim + text_feature_dim + lstm_feature_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, image_features, text_features, lstm_output):
        if lstm_output.dim() == 1:
            lstm_output = lstm_output.unsqueeze(0)
        
        combined = torch.cat([image_features, text_features, lstm_output], dim=1)
        return self.fusion(combined)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss

# Define the hybrid CLIP-LSTM model
class DisDriveClassifierLSTM(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.attention = Attention(256)
        self.fusion = MultiModalFusion()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, images, texts):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
            text_features = self.clip_model.encode_text(texts).float()

        lstm_input = text_features.unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        lstm_output = self.attention(lstm_output.unsqueeze(1)).squeeze(1)

        batch_size = image_features.size(0)
        assert image_features.size(0) == text_features.size(0) == lstm_output.size(0), \
            f"Batch sizes do not match: image {image_features.size(0)}, text {text_features.size(0)}, lstm {lstm_output.size(0)}"

        fused_features = self.fusion(image_features, text_features, lstm_output)
        similarity = self.classifier(fused_features)
        return similarity

# Dataset class
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, text_descriptions, preprocess=preprocess):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.preprocess = preprocess

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)
        text = self.text_descriptions[idx]
        text = clip.tokenize(text).squeeze(0)
        return image, text

    def __len__(self):
        return len(self.image_paths)

# Training function
def train_model(model, train_loader, num_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(alpha=1, gamma=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = GradScaler()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_images, batch_texts in progress_bar:
            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)

            with autocast():
                pos_similarity = model(batch_images, batch_texts)
                neg_texts = batch_texts[torch.randperm(batch_texts.size(0))]
                neg_similarity = model(batch_images, neg_texts)

                pos_labels = torch.ones_like(pos_similarity).to(device)
                neg_labels = torch.zeros_like(neg_similarity).to(device)

                similarities = torch.cat([pos_similarity, neg_similarity])
                labels = torch.cat([pos_labels, neg_labels])

                loss = criterion(similarities, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{total_loss / (batch_images.size(0)):.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

        evaluate_model(model, train_loader)  # Evaluate after each epoch

# Evaluation function for precision-recall and confusion matrix
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, texts in data_loader:
            images = images.to(device)
            texts = texts.to(device)
            outputs = model(images, texts)
            
            preds = torch.round(torch.sigmoid(outputs))  # Convert logits to binary predictions
            all_labels.extend(texts.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plot_precision_recall_curve(precision, recall)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

# Plot precision-recall curve
def plot_precision_recall_curve(precision, recall):
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm):
    disp = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    disp.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    plt.show()

# Create Dataset and DataLoader
# Define the base paths for Camera 1 and Camera 2
camera_1_base = r'D:\Zion 4th Year 1st Sem\THESIS 1 FILES\v2_cam1_cam2_ split_by_driver\Camera 1\train'
camera_2_base = r'D:\Zion 4th Year 1st Sem\THESIS 1 FILES\v2_cam1_cam2_ split_by_driver\Camera 2\train'

# Create paths for directories c0 to c5 in both Camera 1 and Camera 2
image_paths = []

for i in range(6):  # Loop through c0 to c5
    camera_1_dir = os.path.join(camera_1_base, f'c{i}')
    camera_2_dir = os.path.join(camera_2_base, f'c{i}')
    
    # Check if directories exist and add image paths from each
    if os.path.exists(camera_1_dir):
        image_paths += [os.path.join(camera_1_dir, f) for f in os.listdir(camera_1_dir)]
    
    if os.path.exists(camera_2_dir):
        image_paths += [os.path.join(camera_2_dir, f) for f in os.listdir(camera_2_dir)]

# Check the first few image paths
print(image_paths[:5])

# Since we assume a balanced dataset, duplicate text descriptions for now (update if necessary)
text_descriptions = DRIVING_CATEGORIES_PROMPTS * len(image_paths)

dataset = ImageTextDataset(image_paths, text_descriptions)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)  # num_workers=0 to avoid issues on Windows

# Initialize model and start training
model = DisDriveClassifierLSTM(model).to(device)

# Use if __name__ == "__main__": block for multiprocessing compatibility
if __name__ == "__main__":
    train_model(model, train_loader)
