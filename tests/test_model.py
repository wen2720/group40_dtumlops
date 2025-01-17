import pytest
import torch
from src.group40_leaf.model import convnext


def test_convnext_output_shape():
    """
    Test that for a given input with shape (batch_size, 1, 224, 224),
    the model output has shape (batch_size, num_classes).
    """
    num_classes = 99
    model = convnext(num_classes=num_classes)

    # Create a random input of shape (1, 1, 224, 224)
    # batch_size=1, channels=1, height=224, width=224
    x = torch.randn(1, 1, 224, 224)

    # Forward pass
    y = model(x)

    # Check that the output shape is (1, num_classes)
    expected_shape = (1, num_classes)
    assert y.shape == expected_shape, (
        f"Expected output shape to be {expected_shape}, but got {y.shape}"
    )
