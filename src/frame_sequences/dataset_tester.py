import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from hybrid_model import HybridModel

def visualize_augmentations(image_path: str, model: HybridModel, num_samples: int = 5):
    """
    Visualizes original and augmented versions of an image to check for artifacts
    
    Args:
        image_path: Path to test image
        model: Instance of HybridModel containing augmentation pipeline
        num_samples: Number of augmented samples to generate
    """
    # Load image
    original_image = Image.open(image_path)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # Plot original image multiple times in first row
    for i in range(num_samples):
        axes[0, i].imshow(original_image)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
    
    # Plot augmented images in second row
    for i in range(num_samples):
        # Apply augmentation
        augmented = model.augmentation(original_image)
        axes[1, i].imshow(augmented)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Augmented {i+1}')
    
    plt.tight_layout()
    plt.show()

def visualize_sequence_augmentations(sequence_path: str, model: HybridModel, num_frames: int = 5):
    """
    Visualizes original and augmented versions of frames from a sequence
    
    Args:
        sequence_path: Path to sequence folder
        model: Instance of HybridModel containing augmentation pipeline
        num_frames: Number of frames to show from sequence
    """
    frames = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])
    frames = frames[:num_frames]  # Take first n frames
    
    # Create figure
    fig, axes = plt.subplots(2, num_frames, figsize=(15, 6))
    
    # Reset sequence params for consistent augmentation
    sequence_id = os.path.basename(sequence_path)
    model.current_sequence_id = None  # Force reset
    
    # Plot frames
    for i, frame in enumerate(frames):
        # Load original image
        frame_path = os.path.join(sequence_path, frame)
        original_image = Image.open(frame_path)
        
        # Plot original
        axes[0, i].imshow(original_image)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i+1}')
        
        # Apply augmentation
        if model.training:
            if model.current_sequence_id != sequence_id:
                model.current_sequence_id = sequence_id
                model._reset_sequence_params()
            augmented = model._apply_sequence_augmentation(original_image)
        else:
            augmented = original_image
            
        # Plot augmented
        axes[1, i].imshow(augmented)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Augmented {i+1}')
    
    plt.suptitle(f'Sequence: {sequence_id}')
    plt.tight_layout()
    plt.show()

def test_augmentations():
    """Main testing function"""
    # Initialize model
    model = HybridModel()
    model.train()  # Set to training mode to enable augmentations
    
    # Test folder containing one example of each behavior
    test_folder = "./datasets/frame_sequences"  # Update this path
    
    # Test augmentation on one sequence from each behavior class
    for behavior in ['a', 'b', 'c', 'd', 'e', 'f']:
        print(f"\nTesting augmentations for behavior: {behavior}")
        
        # Get first sequence from behavior folder
        behavior_path = os.path.join(test_folder, behavior)
        if os.path.exists(behavior_path):
            sequences = [d for d in os.listdir(behavior_path) 
                       if os.path.isdir(os.path.join(behavior_path, d))]
            
            if sequences:
                test_sequence_path = os.path.join(behavior_path, sequences[0])
                print(f"Testing sequence: {test_sequence_path}")
                visualize_sequence_augmentations(test_sequence_path, model)
            else:
                print(f"No sequences found in: {behavior_path}")
        else:
            print(f"Behavior folder not found: {behavior_path}")

if __name__ == "__main__":
    test_augmentations()