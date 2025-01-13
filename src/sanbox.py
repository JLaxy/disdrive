import torch
from torch.nn.utils.rnn import pad_sequence

# Example: List of tensor sequences of varying lengths
sequences = [torch.rand(10, 512), torch.rand(25, 512), torch.rand(5, 512)]

# Pad sequences
padded_sequences = pad_sequence(sequences, batch_first=True)
# Should be (3, 25, 512)
print("Padded sequences shape:", padded_sequences.shape)
print("test:", torch.stack(sequences))
