import torch
import clip

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# Create a dummy input image tensor
dummy_image = torch.randn(1, 3, 224, 224).to(device)  # Shape: (batch_size, channels, height, width)

# Pass the image through the encoder
image_features = model.encode_image(dummy_image)

# Print the shape of the output
print("Output Shape of CLIP Image Encoder:", image_features.shape)
