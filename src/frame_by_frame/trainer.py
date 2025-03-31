"""Runner Software of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os

TRAINING_DATASET_PATH = "./datasets/frame_by_frame/train"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPOCHS = 20  # Number of Epochs
_LEARNING_RATE = 0.00001  # Learning rate for optimizer in training
_WEIGHT_DECAY = 0.01  # Weight decay for optimizer in training
_TRAINED_MODEL_SAVE_PATH = "./saved_models"
_TO_PREPROCESS_DATA = False


def train_model(dataloader):
    """Train CLIP + LSTM hybrid model with accuracy tracking"""
    CLIP_LSTM.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CLIP_LSTM.parameters(
    ), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(_EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)

        for batch_idx, (b_batch, s_batch) in enumerate(progress_bar):

            b_batch = b_batch.to(_DEVICE)  # Transfer true labels to device
            s_batch = s_batch.clone().detach().to(
                device=_DEVICE, dtype=torch.float32)  # Convert batch of sequence to float32

            optimizer.zero_grad()
            outputs = CLIP_LSTM(s_batch)
            loss = criterion(outputs, b_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == b_batch).sum().item()
            total_samples += b_batch.size(0)

            progress_bar.set_postfix(loss=loss.item())

        epoch_accuracy = (correct_predictions / total_samples) * 100
        avg_loss = running_loss / len(dataloader)

        print(
            f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        scheduler.step(avg_loss)


def save_model_weights(file_name):
    """Saves model weights to disk"""
    torch.save(CLIP_LSTM.state_dict(), os.path.join(
        _TRAINED_MODEL_SAVE_PATH, file_name))  # Save Model Weights

    print(f"Model Saved to disk as '{file_name}'!")


if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()

    dataset = DisDriveDataset(TRAINING_DATASET_PATH,
                              CLIP_LSTM, _TO_PREPROCESS_DATA)

    print(f"sample: {dataset[2500][0]}, {len(dataset[28][1])}")

    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, pin_memory=True)

    train_model(dataloader)

    save_model_weights("disdrive_model.pth")
