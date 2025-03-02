import torch
import torch.nn as nn
import numpy as np
from hybrid_model import HybridModel

# Load trained LSTM model
lstm_model = HybridModel()
lstm_model.load_state_dict(torch.load(
    "./saved_models/disdrive_hybrid_weights.pth", map_location="cpu"))
lstm_model.eval()

# Dummy input for LSTM (batch_size=1, sequence_length=20, feature_dim=512)
dummy_lstm_input = torch.randn(1, 20, 512)

# Export LSTM to ONNX
torch.onnx.export(
    lstm_model,
    dummy_lstm_input,
    "lstm_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["features_sequence"],
    output_names=["prediction"]
)
print("✅ LSTM model converted to ONNX successfully!")
