"""Trainer of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
from tqdm import tqdm
from dataset_splitter import create_train_test_split

_DATASET_PATH = "./datasets/frame_sequences"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPOCHS = 20  # Number of Epochs
_LEARNING_RATE = 0.0001  # Learning rate for optimizer in training
_WEIGHT_DECAY = 0.00001  # Weight decay for optimizer in training
_TRAINED_MODEL_SAVE_PATH = "./saved_models"
_TO_PREPROCESS_DATA = False
_NUM_OF_CLASSES = 6  # Number of classes in the dataset


# def custom_collate_fn(batch):
#     """Custom Collate Function to ensure that batches are in the correct format"""

#     # Collect labels and convert to tensor
#     labels = torch.tensor([item[0] for item in batch])
#     sequences = [item[1] for item in batch]  # Collect sequences of images
#     return labels, sequences


def __dataloader_debug(dataloader):
    """Function used to debug the dataloader"""

    print("Debugging dataloader...\n")

    for behavior_batch, sequence_batch in dataloader:
        # Print behavior batch
        print(f"behavior batch type: {type(behavior_batch)}")
        print(f"behavior batch length: {len(behavior_batch)}")
        print(f"behavior batch: {behavior_batch}")
        print(f"behavior batch shape: {behavior_batch.shape}")

        # Print sequence batch
        print(f"\nsequence batch type: {type(sequence_batch)}")
        print(f"sequence batch length: {len(sequence_batch)}")
        # print(f"image shape: {image.shape}")

        # Print single instance of sequence
        for sequence in sequence_batch:
            print(type(sequence))
            print(len(sequence))
            print(sequence.shape)
            print(type(sequence[0]))
            print(sequence[0].shape)
            break

        break


def calculate_class_metrics(true_labels, predicted_labels, num_classes):
    """Calculate per-class accuracy"""
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for t, p in zip(true_labels, predicted_labels):
        if t == p:
            class_correct[t] += 1
        class_total[t] += 1
    
    # Avoid division by zero
    class_accuracies = torch.where(
        class_total != 0, 
        100.0 * class_correct / class_total,
        torch.tensor(0.0)
    )
    
    return class_accuracies


def train_model(train_dataloader, val_dataloader):
    """Trains Hybrid Model using dataset with validation"""
    CLIP_LSTM.train()

    # Calculate class weights to handle imbalanced data
    class_counts = torch.zeros(_NUM_OF_CLASSES)
    for b_batch, _ in train_dataloader:
        class_counts += torch.bincount(b_batch, minlength=_NUM_OF_CLASSES)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(_DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        CLIP_LSTM.parameters(),
        lr=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=True
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(_EPOCHS):
        CLIP_LSTM.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Add tracking for per-class metrics
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for b_batch, s_batch in progress_bar:
            b_batch = b_batch.to(_DEVICE)
            s_batch = s_batch.to(_DEVICE, dtype=torch.float32)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = CLIP_LSTM(s_batch)
                loss = criterion(output, b_batch)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                CLIP_LSTM.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (predicted == b_batch).sum().item()
            total_samples += b_batch.size(0)

            # Store predictions and labels for class metrics
            all_predictions.extend(predicted.cpu())
            all_labels.extend(b_batch.cpu())

            progress_bar.set_postfix(
                loss=loss.item(),
                acc=f"{100. * correct_predictions/total_samples:.2f}%"
            )

        # Calculate per-class accuracies for training
        train_class_accuracies = calculate_class_metrics(
            torch.tensor(all_labels), 
            torch.tensor(all_predictions), 
            _NUM_OF_CLASSES
        )

        # Validation phase with per-class metrics
        val_loss, val_accuracy, val_class_accuracies = validate_model(
            CLIP_LSTM, val_dataloader, criterion)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model_weights("best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Print epoch results with per-class metrics
        print(f"\nEpoch {epoch+1}/{_EPOCHS}:")
        print(f"Training Loss: {running_loss/len(train_dataloader):.4f}")
        print(f"Training Accuracy: {100. * correct_predictions/total_samples:.2f}%")
        print("\nPer-class Training Accuracies:")
        for i, acc in enumerate(train_class_accuracies):
            print(f"Class {i}: {acc:.2f}%")
            
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print("\nPer-class Validation Accuracies:")
        for i, acc in enumerate(val_class_accuracies):
            print(f"Class {i}: {acc:.2f}%")


def validate_model(model, val_dataloader, criterion):
    """Validates model performance on validation set"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for b_batch, s_batch in val_dataloader:
            b_batch = b_batch.to(_DEVICE)
            s_batch = s_batch.to(_DEVICE, dtype=torch.float32)

            outputs = model(s_batch)
            loss = criterion(outputs, b_batch)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += b_batch.size(0)
            correct += predicted.eq(b_batch).sum().item()
            
            # Store predictions and labels for class metrics
            all_predictions.extend(predicted.cpu())
            all_labels.extend(b_batch.cpu())

    # Calculate per-class accuracies
    class_accuracies = calculate_class_metrics(
        torch.tensor(all_labels), 
        torch.tensor(all_predictions), 
        _NUM_OF_CLASSES
    )

    return val_loss / len(val_dataloader), 100. * correct / total, class_accuracies


def save_model_weights(file_name):
    """Saves model weights to disk"""
    torch.save(CLIP_LSTM.state_dict(), os.path.join(
        _TRAINED_MODEL_SAVE_PATH, file_name))  # Save Model Weights

    print(f"Model Saved to disk as '{file_name}'!")


if __name__ == "__main__":
    CLIP_LSTM = HybridModel()
    CLIP_LSTM.to(_DEVICE)

    full_dataset = DisDriveDataset(
        _DATASET_PATH, CLIP_LSTM, _TO_PREPROCESS_DATA)

    # Split into train, validation and test sets
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size])

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,  # Reduced batch size for better stability
        pin_memory=True,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        pin_memory=True,
        shuffle=False
    )

    train_model(train_dataloader, val_dataloader)
    save_model_weights("final_model.pth")
