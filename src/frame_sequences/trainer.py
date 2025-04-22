"""Trainer of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from multiprocessing import freeze_support
from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
from tqdm import tqdm
from dataset_splitter import create_train_test_split
import matplotlib.pyplot as plt

_DATASET_PATH = "./datasets/frame_sequences"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPOCHS = 5  # Number of Epochs
_LEARNING_RATE = 0.0001  # Learning rate for optimizer in training
_WEIGHT_DECAY = 0.00001  # Weight decay for optimizer in training
_TRAINED_MODEL_SAVE_PATH = "./saved_models"
_TO_PREPROCESS_DATA = False
_TO_USE_PRECOMPUTED = False  # Use precomputed features or not
_NUM_OF_CLASSES = 6  # Number of classes in the dataset

stage1_thresholds = {
    "adapter": (1e-3, 5.0),
    "lstm": (1e-3, 5.0),
    "view_attention": (1e-3, 5.0),
    "view_embedding": (1e-3, 5.0),
    "temporal_attention": (1e-3, 5.0),
    "fc": (1e-2, 10.0),
    "clip_model": (0.0, 0.0),  # Should be frozen, no gradients
}

stage2_thresholds = {
    "adapter": (5e-4, 2.0),
    "lstm": (5e-4, 2.0),
    "view_attention": (5e-4, 2.0),
    "view_embedding": (5e-4, 2.0),
    "temporal_attention": (5e-4, 2.0),
    "fc": (1e-3, 5.0),
    "clip_model": (1e-4, 1.0),  # Partially unfrozen, smaller gradients
}

stage3_thresholds = {
    "adapter": (1e-4, 1.0),
    "lstm": (1e-4, 1.0),
    "view_attention": (1e-4, 1.0),
    "view_embedding": (1e-4, 1.0),
    "temporal_attention": (1e-4, 1.0),
    "fc": (5e-4, 2.0),
    "clip_model": (5e-5, 0.5),  # All unfrozen, very small gradients
}

layer_thresholds = {
    "adapter": (1e-3, 1.0),
    "lstm": (1e-3, 1.0),
    "classifier": (1e-2, 10.0),
    "clip_model": (0.0, 1e-3),  # If frozen, otherwise raise upper bound
}

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


def monitor_gradients(model, epoch, thresholds):
    """
    Monitor gradient norms with stage-specific thresholds

    Args:
        model: The model being trained
        epoch: Current epoch number (for display purposes)
        thresholds: Dictionary of min/max thresholds by layer name
    """
    print(f"\n🔍 Gradient Monitoring for Epoch {epoch+1}")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            # Find matching threshold
            matched = False
            for key, (min_th, max_th) in thresholds.items():
                if key in name:
                    status = "✅ OK"
                    if grad_norm < min_th:
                        status = "⚠️ Too Low"
                    elif grad_norm > max_th:
                        status = "🚨 Too High"
                    print(
                        f"{name:50} | Grad Norm: {grad_norm:.6f} | Range: [{min_th}, {max_th}] -> {status}")
                    matched = True
                    break

            # If no specific threshold was found
            if not matched:
                print(
                    f"{name:50} | Grad Norm: {grad_norm:.6f} | No specific threshold")
        else:
            # Check if parameter should have gradients based on requires_grad status
            if param.requires_grad:
                print(f"{name:50} | ⚠️ No Gradient (but requires_grad=True)")
            else:
                print(f"{name:50} | 🧊 Frozen (requires_grad=False)")


def train_model(train_dataloader, val_dataloader):
    """Trains Hybrid Model using dataset with validation"""

    # Calculate class weights to handle imbalanced data
    class_counts = torch.zeros(_NUM_OF_CLASSES)
    for b_batch, _, _ in train_dataloader:
        class_counts += torch.bincount(b_batch, minlength=_NUM_OF_CLASSES)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(_DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Stage 1: Train only new components (CLIP remains frozen)
    print("===== Starting Stage 1: Training adapter, LSTM and classification layers =====")
    set_frozen_status(CLIP_LSTM, stage=0)
    stage1_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, CLIP_LSTM.parameters()),
        lr=2e-3,
        weight_decay=_WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    train_stage(CLIP_LSTM, train_dataloader, val_dataloader, criterion,
                stage1_optimizer, max_epochs=10, stage_name="Stage 1", thresholds=stage1_thresholds, patience=3)

    # Stage 2: Unfreeze deeper CLIP layers
    print("===== Starting Stage 2: Fine-tuning with partial CLIP unfreezing =====")
    set_frozen_status(CLIP_LSTM, stage=1)
    stage2_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, CLIP_LSTM.parameters()),
        lr=5e-4,
        weight_decay=_WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    train_stage(CLIP_LSTM, train_dataloader, val_dataloader, criterion,
                stage2_optimizer, max_epochs=10, stage_name="Stage 2", thresholds=stage2_thresholds, patience=3)

    # Stage 3: Unfreeze all layers
    print("===== Starting Stage 3: Fine-tuning with all layers unfrozen =====")
    set_frozen_status(CLIP_LSTM, stage=2)
    stage3_optimizer = torch.optim.AdamW(
        CLIP_LSTM.parameters(),
        lr=1e-4,
        weight_decay=_WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    train_stage(CLIP_LSTM, train_dataloader, val_dataloader, criterion,
                stage3_optimizer, max_epochs=10, stage_name="Stage 3", thresholds=stage3_thresholds, patience=3)


def set_frozen_status(model, stage):
    """
    Set which layers should be frozen based on training stage
    stage: 0 = only CLIP frozen, 1 = partial CLIP unfrozen, 2 = all unfrozen
    """
    # First freeze everything in CLIP
    for param in model.clip_model.parameters():
        param.requires_grad = False

    # Make sure adapter, LSTM, etc. are always trainable
    for name, param in model.named_parameters():
        if "clip_model" not in name:
            param.requires_grad = True

    if stage >= 1:
        # Unfreeze final transformer blocks of CLIP's vision encoder
        blocks_to_unfreeze = 2
        for i, block in enumerate(reversed(model.clip_model.visual.transformer.resblocks)):
            if i < blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True

    if stage >= 2:
        # Unfreeze all CLIP layers
        for param in model.clip_model.parameters():
            param.requires_grad = True

    # Print trainable parameters count
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}% of total)")


def train_stage(model, train_dataloader, val_dataloader, criterion, optimizer,
                max_epochs, stage_name, thresholds, patience=3):
    """
    Trains for a specific stage with the given optimizer
    """
    model.train()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')
    patience_counter = 0

    # Add lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(
            train_dataloader, desc=f"{stage_name} - Epoch {epoch+1}")

        for b_batch, s_batch, v_batch in progress_bar:
            b_batch = b_batch.to(_DEVICE)
            s_batch = s_batch.to(_DEVICE, dtype=torch.float32)
            v_batch = v_batch.to(_DEVICE, dtype=torch.float32)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = model(s_batch, v_batch)
                loss = criterion(output, b_batch)

            scaler.scale(loss).backward()

            # Unscale and check gradients
            scaler.unscale_(optimizer)
            # Clip & step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"❌ No grad for {name}")

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (predicted == b_batch).sum().item()
            total_samples += b_batch.size(0)

            progress_bar.set_postfix(
                loss=loss.item(),
                acc=f"{100. * correct_predictions/total_samples:.2f}%"
            )

        monitor_gradients(model, epoch, thresholds)

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_loss, val_accuracy, val_class_accuracies = validate_model(
            model, val_dataloader, criterion)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model_weights(
                f"best_model_{stage_name.replace(' ', '_')}.pth")
        else:
            patience_counter += 1

        print(f"Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_weights(
                f"best_model_{stage_name.replace(' ', '_')}.pth")

        # Print epoch results
        print(f"\n{stage_name} - Epoch {epoch+1}/{max_epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(
            f"Training Accuracy: {100. * correct_predictions/total_samples:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")


def validate_model(model, val_dataloader, criterion):
    """Validates model performance on validation set"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for b_batch, s_batch, v_batch in val_dataloader:
            b_batch = b_batch.to(_DEVICE)
            s_batch = s_batch.to(_DEVICE, dtype=torch.float32)
            v_batch = v_batch.to(_DEVICE, dtype=torch.float32)

            outputs = model(s_batch, v_batch)
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


def train_model_frozen(train_dataloader, val_dataloader):
    """Trains Hybrid Model using dataset with validation"""
    CLIP_LSTM.train()

    set_frozen_status(CLIP_LSTM, stage=0)

    # Calculate class weights to handle imbalanced data
    class_counts = torch.zeros(_NUM_OF_CLASSES)
    for b_batch, _, _ in train_dataloader:
        class_counts += torch.bincount(b_batch, minlength=_NUM_OF_CLASSES)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(_DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
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

    # Add lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(_EPOCHS):
        CLIP_LSTM.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Add tracking for per-class metrics
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for b_batch, s_batch, v_batch in progress_bar:
            b_batch = b_batch.to(_DEVICE)
            s_batch = s_batch.to(_DEVICE, dtype=torch.float32)
            v_batch = v_batch.to(_DEVICE, dtype=torch.float32)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = CLIP_LSTM(s_batch, v_batch)
                loss = criterion(output, b_batch)

            scaler.scale(loss).backward()

            # Unscale and check gradients
            scaler.unscale_(optimizer)
            # Clip & step
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

        monitor_gradients(CLIP_LSTM, epoch, layer_thresholds)

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Calculate per-class accuracies for training
        train_class_accuracies = calculate_class_metrics(
            torch.tensor(all_labels),
            torch.tensor(all_predictions),
            _NUM_OF_CLASSES
        )

        # Validation phase with per-class metrics
        val_loss, val_accuracy, val_class_accuracies = validate_model(
            CLIP_LSTM, val_dataloader, criterion)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🔁 Current Learning Rate: {current_lr:.6f}")

        # Optional: Warn if learning rate is too low
        if current_lr <= 1e-6:
            print("⚠️ Learning rate has reached the minimum threshold. Consider reviewing model capacity or data quality.")

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

        # Print epoch results with losses
        print(f"\nEpoch {epoch+1}/{_EPOCHS}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(
            f"Training Accuracy: {100. * correct_predictions/total_samples:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Print per-class metrics
        print("\nPer-class Training Accuracies:")
        for i, acc in enumerate(train_class_accuracies):
            print(f"Class {i}: {acc:.2f}%")

        print("\nPer-class Validation Accuracies:")
        for i, acc in enumerate(val_class_accuracies):
            print(f"Class {i}: {acc:.2f}%")

        # Optional: Plot losses every few epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == _EPOCHS - 1:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Time')
            plt.legend()
            plt.show()


if __name__ == "__main__":
    CLIP_LSTM = HybridModel(_TO_USE_PRECOMPUTED)
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
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        pin_memory=True,
        shuffle=False,
    )

    train_model(train_dataloader, val_dataloader)
    save_model_weights("final_model.pth")
