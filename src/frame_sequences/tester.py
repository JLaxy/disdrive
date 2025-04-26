"""Trainer of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DATASET_PATH = "./datasets/frame_sequences"
_TRAINED_MODEL_SAVE_PATH = "./saved_models/final_model.pth"
_TO_PREPROCESS_DATA = False
_BEHAVIOR_LABELS = ["Safe Driving",
                    "Texting",
                    "Talking using Phone",
                    "Drinking",
                    "Head Down",
                    "Look Behind"]


def test_model(dataloader):
    """Tests Hybrid Model according to metrics"""
    print("Testing Model...")
    CLIP_LSTM.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        # b_batch: Batch of Behavior Labels
        # s_batch: Batch of Sequences of frames
        for b_batch, s_batch, v_batch in dataloader:
            # Move labels and sequences to device
            b_batch = b_batch.to(_DEVICE)
            # Convert batch of sequence to float32
            s_batch = s_batch.clone().detach().to(device=_DEVICE, dtype=torch.float32)
            v_batch = v_batch.to(_DEVICE)

            output = CLIP_LSTM(s_batch, v_batch)
            prediction = torch.argmax(output, dim=1)

            true_labels.extend(b_batch.cpu().numpy())
            pred_labels.extend(prediction.cpu().numpy())

    print("Done! printing results....")

    # Display report
    print(classification_report(true_labels, pred_labels,
          target_names=_BEHAVIOR_LABELS))

    print(f"Model Accuracy: {accuracy_score(true_labels, pred_labels)}")

    cm = confusion_matrix(true_labels, pred_labels)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=_BEHAVIOR_LABELS)
    # Rotate labels for readability
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # Initialize model
    CLIP_LSTM: HybridModel = HybridModel(use_precomputed=True)
    # Load Weights
    CLIP_LSTM.load_state_dict(torch.load(_TRAINED_MODEL_SAVE_PATH))
    # Move Hybrid Model to device
    CLIP_LSTM.to(_DEVICE)

    full_dataset = DisDriveDataset(_DATASET_PATH,
                                   CLIP_LSTM, _TO_PREPROCESS_DATA)

    # Split dataset like in trainer.py
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Create splits
    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size])

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Initialize Dataloader for test set
    dataloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False, pin_memory=True)

    test_model(dataloader)  # Test
