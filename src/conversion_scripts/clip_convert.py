import torch
import clip
import torch.onnx

# Load CLIP model
# Use a smaller CLIP variant if needed
model, preprocess = clip.load("ViT-B/16", device="cpu")
model.eval()

# Example input (1 image, 3 color channels, 224x224 pixels)
dummy_input = torch.randn(1, 3, 224, 224)

# Convert to ONNX
torch.onnx.export(
    model.visual,
    dummy_input,
    "clip_visual.onnx",
    export_params=True,
    opset_version=14,
    input_names=["input"],
    output_names=["features"]
)
