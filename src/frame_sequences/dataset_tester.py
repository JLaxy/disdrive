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

def test_augmentations():
    """Main testing function"""
    # Initialize model
    model = HybridModel()
    model.train()  # Set to training mode to enable augmentations
    
    # Test folder containing one example of each behavior
    test_folder = "./datasets/test"  # Update this path
    
    # Test augmentation on one example from each behavior class
    for behavior in ['a', 'b', 'c', 'd', 'e', 'f']:
        print(f"\nTesting augmentations for behavior: {behavior}")
        
        # Get first image from behavior folder
        behavior_path = os.path.join(test_folder, behavior)
        if os.path.exists(behavior_path):
            test_sequence = os.listdir(behavior_path)[0]
            test_image = os.path.join(behavior_path, test_sequence)
            
            if os.path.exists(test_image):
                print(f"Testing image: {test_image}")
                visualize_augmentations(test_image, model)
            else:
                print(f"Test image not found: {test_image}")
        else:
            print(f"Behavior folder not found: {behavior_path}")

if __name__ == "__main__":
    test_augmentations()