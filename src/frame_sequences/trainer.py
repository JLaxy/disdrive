"""Trainer of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
from tqdm import tqdm

TRAINING_DATASET_PATH = "./datasets/frame_sequences/train"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPOCHS = 20  # Number of Epochs
_LEARNING_RATE = 0.0001  # Learning rate for optimizer in training
_WEIGHT_DECAY = 0.0001  # Weight decay for optimizer in training
_TRAINED_MODEL_SAVE_PATH = "./saved_models"
_TO_PREPROCESS_DATA = True
_TO_DANGER_BOOST = True  # Boosting the danger level of the data
_NUM_OF_CLASSES = 8  # Number of classes in the dataset


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


def train_model(dataloader):
    """Trains Hybrid Model using dataset"""
    CLIP_LSTM.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CLIP_LSTM.parameters(
    ), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)

    # Add these variables before your training loop
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    class_correct = torch.zeros(_NUM_OF_CLASSES).to(
        _DEVICE)  # Number of correct predictions per class
    class_total = torch.zeros(_NUM_OF_CLASSES).to(
        _DEVICE)  # Total number of samples per class

    for epoch in range(_EPOCHS):  # Cycle each Epoch

        # Initialize progress bar
        progress_bar = tqdm(
            dataloader, desc=f"Training Epoch {epoch+1}", leave=True)

        running_loss = 0.0  # Contains loss value of model during training
        correct_predictions = 0  # Number of correct predictions
        total_samples = 0  # Total number of samples
        class_distrib = torch.zeros(_NUM_OF_CLASSES)  # Distribution of batches

        # b_batch: Batch of Behavior Labels
        # s_batch: Batch of Sequences of frames
        for batch_idx, (b_batch, s_batch) in enumerate(progress_bar):

            class_distrib += torch.bincount(b_batch,
                                            minlength=_NUM_OF_CLASSES)

            b_batch = b_batch.to(_DEVICE)  # Transfer true labels to device
            s_batch = s_batch.clone().detach().to(
                device=_DEVICE, dtype=torch.float32)  # Convert batch of sequence to float32

            # Clear gradients
            optimizer.zero_grad()
            # Forward pass (make prediction)
            output = CLIP_LSTM(s_batch)
            # Compute difference of true and predicted values
            loss = criterion(output, b_batch)
            # Backward pass; let model learn
            loss.backward()
            # Readjust weights of model to apply learning
            optimizer.step()

            running_loss += loss.item()

            # Computing accuracy
            predicted = torch.argmax(output, dim=1)  # Get predicted behavior
            # Add if correct
            correct_predictions += (predicted == b_batch).sum().item()
            total_samples += b_batch.size(0)

            progress_bar.set_postfix(loss=loss.item())

            for i in range(_NUM_OF_CLASSES):
                mask = (b_batch == i)
                class_correct[i] += (predicted[mask] == b_batch[mask]).sum()
                class_total[i] += mask.sum()

        # Add inside your epoch loop after calculating loss
        if running_loss/len(dataloader) < best_loss:
            best_loss = running_loss/len(dataloader)
            patience_counter = 0
            save_model_weights("best_disdrive_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        epoch_accuracy = (correct_predictions / total_samples) * 100

        print(
            f"Epoch [{epoch+1}/{_EPOCHS}], Loss: {running_loss/len(dataloader)}, Accuracy: {epoch_accuracy:.2f}%")

        avg_class_dist = class_distrib / len(dataloader)  # Average per batch
        print("Class distribution per batch:")
        for class_idx, count in enumerate(avg_class_dist):
            print(f"  Class {class_idx}: {count:.1f} samples")

        print("\nPer-Class Accuracy:")
        for i in range(_NUM_OF_CLASSES):
            if class_total[i] > 0:
                print(
                    f"Class {i}: {100 * class_correct[i] / class_total[i]:.1f}%")


def save_model_weights(file_name):
    """Saves model weights to disk"""
    torch.save(CLIP_LSTM.state_dict(), os.path.join(
        _TRAINED_MODEL_SAVE_PATH, file_name))  # Save Model Weights

    print(f"Model Saved to disk as '{file_name}'!")


if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()
    CLIP_LSTM.to(_DEVICE)  # Move Hybrid Model to device

    dataset = DisDriveDataset(TRAINING_DATASET_PATH,
                              CLIP_LSTM, _TO_PREPROCESS_DATA, _TO_DANGER_BOOST)

    weights = dataset.get_weights()
    print(f"Length: {len(weights)}")
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True)

    dataloader = DataLoader(dataset, batch_size=32,
                            pin_memory=True, sampler=sampler)

    # __dataloader_debug(dataloader)

    train_model(dataloader)
    save_model_weights("boosted_disdrive_model.pth")
