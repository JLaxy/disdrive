import torch
from hybrid_model import HybridModel, DisDriveDataset

_DATASET_PATH = "./datasets/frame_sequences"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPOCHS = 5  # Number of Epochs
_LEARNING_RATE = 0.0001  # Learning rate for optimizer in training
_WEIGHT_DECAY = 0.00001  # Weight decay for optimizer in training
_TRAINED_MODEL_SAVE_PATH = "./saved_models"
_TO_PREPROCESS_DATA = False
_TO_USE_PRECOMPUTED = False  # Use precomputed features or not
_NUM_OF_CLASSES = 6  # Number of classes in the dataset


def set_frozen_status(model, stage):
    """
    Set which layers should be frozen based on training stage
    stage: 0 = only CLIP frozen, 1 = partial CLIP unfrozen, 2 = all unfrozen
    """
    # First set requires_grad=False for all CLIP parameters
    for param in model.clip_model.parameters():
        param.requires_grad = False

    # Make sure non-CLIP components are always trainable
    for name, param in model.named_parameters():
        if "clip_model" not in name:
            param.requires_grad = True

    # Stage-specific unfreezing
    if stage >= 2:  # Stage 3: Unfreeze everything
        print("Stage 3: Unfreezing all CLIP layers")
        for param in model.clip_model.parameters():
            param.requires_grad = True

    elif stage >= 1:  # Stage 2: Partial unfreeze
        print("Stage 2: Unfreezing last transformer blocks")
        blocks_to_unfreeze = 2
        for i, block in enumerate(reversed(model.clip_model.visual.transformer.resblocks)):
            if i < blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True

    # Print trainable parameter count
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}% of total)")


def check_trainable(model):
    # Print trainable parameters count
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}% of total)")


def check_gradients(model):
    """Check which parameters are trainable and have gradients"""
    print("\nParameter Status:")
    trainable_count = 0
    grad_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            has_grad = param.grad is not None
            if has_grad:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                status = f"✅ Trainable, grad_norm: {grad_norm:.4e}"
            else:
                status = "⚠️ Trainable, but no gradient"
        else:
            status = "❌ Frozen"

        print(f"{name}: {status}")

    print(
        f"\nSummary: {grad_count}/{trainable_count} trainable parameters have gradients")
    return grad_count > 0


def test_gradients(model):
    """
    Test if gradients are being created by passing dummy data through the model
    """
    # model.train()  # Set to training mode

    # Create dummy input data
    batch_size = 32
    seq_length = 20

    # Process frames through CLIP first
    dummy_frames = torch.randn(
        batch_size * seq_length, 3, 224, 224).to(_DEVICE)

    # Extract CLIP features
    clip_features = model.clip_model.encode_image(
        dummy_frames)  # [batch_size*seq_length, 512]
    # Reshape to [batch_size, seq_length, 512]
    clip_features = clip_features.view(batch_size, seq_length, -1)

    dummy_views = torch.randint(0, 2, (batch_size,)).to(
        _DEVICE)  # 0=front, 1=side view
    dummy_labels = torch.randint(0, _NUM_OF_CLASSES, (batch_size,)).to(_DEVICE)

    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Forward pass
    optimizer.zero_grad()
    # Pass processed features instead of raw frames
    outputs = model(clip_features, dummy_views)
    loss = criterion(outputs, dummy_labels)

    print(f"\nInitial loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_grads = check_gradients(model)
    print("\nChecking gradients after backward pass:")
    grad_status = {}
    total_params = 0
    params_with_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            has_grad = param.grad is not None
            if has_grad:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                grad_status[name] = f"✅ (grad norm: {grad_norm:.4e})"
            else:
                grad_status[name] = "❌ No gradient"
            print(f"{name}: {grad_status[name]}")

    grad_percentage = (params_with_grad / total_params *
                       100) if total_params > 0 else 0
    print(
        f"\nGradient Summary: {params_with_grad}/{total_params} parameters have gradients ({grad_percentage:.1f}%)")

    return params_with_grad > 0


if __name__ == "__main__":
    print("This is a test script for debugging gradients.")
    CLIP_LSTM = HybridModel(_TO_USE_PRECOMPUTED)
    CLIP_LSTM.to(_DEVICE)

    CLIP_LSTM.train()

    set_frozen_status(CLIP_LSTM, 2)
    check_trainable(CLIP_LSTM)

    print("\nTesting gradient creation...")
    has_grads = test_gradients(CLIP_LSTM)
    print(
        f"\nGradient test result: {'✅ Gradients are being created' if has_grads else '❌ No gradients created'}")
