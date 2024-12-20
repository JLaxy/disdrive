import torch
import clip
import glob
import numpy as np
from PIL import Image

# Selecting which device to use; CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# Prompts for distracted driving behaviors
DRIVING_CATEGORIES_PROMPTS = [
    "a photo of a person driving safely with both hands on the steering wheel",
    "a photo of a person texting using their mobile phone while driving",
    "a photo of a person talking on the phone while driving",
    "a photo of a person operating the car radio while driving",
    "a photo of a person drinking while driving",
    "a photo of a person reaching for an object behind them while driving",
    "a photo of a person trying to fix their hair while driving",
]
# Tokenize the prompts
tokenized_driving_categories_prompts = clip.tokenize(
    DRIVING_CATEGORIES_PROMPTS).to(device)


def get_image_embeddings(image):
    """Returns the image embeddings for a single image; resize the image to 336x336 before processing."""
    # Preprocess the image
    processed_image = preprocess(image).unsqueeze(0).to(device)
    # Get the image embeddings
    with torch.no_grad():
        image_embeddings = model.encode_image(processed_image)
    return image_embeddings


def get_text_embeddings(text: str):
    """Returns the text embeddings for a single text."""
    # Process the text
    processed_text = clip.tokenize(text).to(device)
    # Get the text embeddings
    with torch.no_grad():
        text_embeddings = model.encode_text(processed_text)
    return text_embeddings


def get_distracted_driving_behavior_probabilities(image):
    """Returns its probablities for each distracted driving behavior."""
    # Open image then automatically resize it to 336x336
    image = preprocess(image).unsqueeze(0).to(device)

    # Calculate probabilities
    with torch.no_grad():
        logits_per_image, logits_per_text = model(
            image, tokenized_driving_categories_prompts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        for label, prob in zip(DRIVING_CATEGORIES_PROMPTS, np.ravel(probs)):
            print(f"{label}: {prob:.2%}")

        return np.ravel(probs)


# print(get_distracted_driving_behavior_probabilities(
#     Image.open("images/76385.jpg")))



