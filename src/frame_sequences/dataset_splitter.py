from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch
import os
import json


def save_split_indices(train_indices, test_indices, save_path="./split_indices.json"):
    with open(save_path, 'w') as f:
        json.dump({
            'train': train_indices.tolist(),
            'test': test_indices.tolist()
        }, f)


def load_split_indices(save_path="./split_indices.json"):
    with open(save_path, 'r') as f:
        indices = json.load(f)
    return np.array(indices['train']), np.array(indices['test'])


def create_train_test_split(dataset, test_size=0.2, random_state=28, save_indices=True):
    """
    Creates a stratified train/test split ensuring all categories are represented

    Args:
        dataset: The full dataset
        test_size: Proportion of dataset to include in test split (default 0.2)
        random_state: Random seed for reproducibility
        save_indices: Whether to save the split indices to a file (default True)
    """

    # Get all labels from dataset
    labels = [dataset[i][0] for i in range(len(dataset))]
    labels = np.array(labels)

    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print("\nInitial class distribution:")
    print("Class\tCount\tPercentage")
    print("-" * 40)
    for label, count in zip(unique_labels, label_counts):
        percentage = (count / len(labels)) * 100
        print(f"{label}\t{count}\t{percentage:.2f}%")

    # Verify we have all 6 categories
    if len(unique_labels) != 6:
        missing = set(range(6)) - set(unique_labels)
        print(f"\nWarning: Missing categories: {missing}")

    # Create stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(
        splitter.split(np.zeros(len(labels)), labels))

    # Print distribution information
    train_dist = np.bincount(labels[train_indices])
    test_dist = np.bincount(labels[test_indices])

    print("\nClass distribution in splits:")
    print("Class\tTrain\tTest\tTrain%\tTest%")
    print("-" * 50)
    for class_idx in range(len(train_dist)):
        train_percent = (train_dist[class_idx] / len(train_indices)) * 100
        test_percent = (test_dist[class_idx] / len(test_indices)) * 100
        print(
            f"{class_idx}\t{train_dist[class_idx]}\t{test_dist[class_idx]}\t{train_percent:.1f}%\t{test_percent:.1f}%")

    # Create train and test datasets using indices
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    if save_indices:
        save_split_indices(train_indices, test_indices)

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    dataset_directory = "./datasets/frame_sequences"

    # Assuming you have a dataset class that can be instantiated
    from hybrid_model import DisDriveDataset, HybridModel

    model = HybridModel()  # Replace with actual model initialization if needed

    dataset = DisDriveDataset(dataset_directory, model, to_get_features=False)

    train_dataset, test_dataset = create_train_test_split(
        dataset, test_size=0.2, random_state=42)
