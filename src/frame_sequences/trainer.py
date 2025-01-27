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


if __name__ == "__main__":
    CLIP_LSTM: HybridModel = HybridModel()

    dataset = DisDriveDataset(TRAINING_DATASET_PATH, CLIP_LSTM)

    print(f"sample: {dataset[28][0]}, \n{len(dataset[28][1])}")

    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True)

    for behavior, sequence in dataloader:
        print(f"behavior type: {type(behavior)}")
        print(f"behavior size: {len(behavior)}")
        print(f"behavior: {behavior}")
        print(f"behavior shape: {behavior.shape}")

        print(f"sequence type: {type(sequence)}")
        print(f"sequence length: {len(sequence)}")
        # print(f"image shape: {image.shape}")

        for seq in sequence:
            print(type(seq))
            print(len(seq))
            print(seq[0].shape)
            break

        # output = CLIP_LSTM(image)

        # print(output)
        # print(type(output))
        # print(output.shape)

        break
