import torch
from fvcore.nn import FlopCountAnalysis
from frame_sequences.hybrid_model_original import HybridModel

_TRAINED_MODEL_SAVE_PATH = "./saved_models/disdrive_hybrid_weights.pth"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load your trained model
model: HybridModel = HybridModel()  # Replace with your actual model instance
model.load_state_dict(torch.load(_TRAINED_MODEL_SAVE_PATH))
model.to(_DEVICE)
model.eval()  # Set to evaluation mode

# Dummy input: Replace with actual input shape (batch_size, sequence_length, feature_dim)
# Example input for LSTM with 20 frames
dummy_input = torch.randn(1, 20, 512).to(_DEVICE)

# Compute FLOPS
flops = FlopCountAnalysis(model, dummy_input)

# Print results
print(f"Total FLOPS: {flops.total()} FLOPS")
print(f"Total FLOPS in GFLOPS: {flops.total() / 1e9} GFLOPS")
