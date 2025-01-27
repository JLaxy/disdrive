"""Runner Software of the Distracted Driving Behavior Detector using CLIP and LSTM"""

from hybrid_model import DisDriveDataset, HybridModel
from torch.utils.data import DataLoader
import torch

TRAINING_DATASET_PATH = "./datasets/frame_sequences/train"


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
            print(sequence[0].shape)
            break

        break


if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()

    dataset = DisDriveDataset(TRAINING_DATASET_PATH, CLIP_LSTM)

    print(f"sample: {dataset[28][0]}, {len(dataset[28][1])}")

    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True)

    __dataloader_debug(dataloader)
