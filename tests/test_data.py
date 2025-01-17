# tests/test_data.py

import os
import pytest
import torch
import pandas as pd
from src.group40_leaf.data import LeafDataset, get_dataloaders  # Adjust according to the actual path

TRAIN_CSV_PATH = "leaf-classification/train.csv"
TEST_CSV_PATH = "leaf-classification/test.csv"
IMAGE_DIR = "leaf-classification/images_full"

# Expected number of samples (only for training set)
EXPECTED_N_TRAIN = 990

@pytest.mark.skipif(
    not (
        os.path.exists(IMAGE_DIR) and
        os.path.exists(TRAIN_CSV_PATH) and
        os.path.exists(TEST_CSV_PATH)
    ),
    reason="Data files not found"
)
def test_train_dataset_length():
    """
    Test whether the training LeafDataset has the correct number of samples.
    """
    train_dataset = LeafDataset(
        image_dir=IMAGE_DIR,
        csv_path=TRAIN_CSV_PATH,
        transform=None
    )
    assert len(train_dataset) == EXPECTED_N_TRAIN, (
        f"Training dataset length {len(train_dataset)} does not match expected {EXPECTED_N_TRAIN}"
    )

@pytest.mark.skipif(
    not (
        os.path.exists(IMAGE_DIR) and
        os.path.exists(TRAIN_CSV_PATH) and
        os.path.exists(TEST_CSV_PATH)
    ),
    reason="Data files not found"
)
def test_all_labels_present_in_train():
    """
    Test whether all possible labels are present in the training dataset (requires 'species' column).
    """
    train_dataset = LeafDataset(
        image_dir=IMAGE_DIR,
        csv_path=TRAIN_CSV_PATH,
        transform=None
    )
    labels = set()
    for _, label in train_dataset:
        labels.add(label)
    expected_num_classes = len(train_dataset.species_to_label)
    actual_num_classes = len(labels)
    assert actual_num_classes == expected_num_classes, (
        f"Number of unique labels in training dataset {actual_num_classes} "
        f"does not match expected {expected_num_classes}"
    )

@pytest.mark.skipif(
    not (
        os.path.exists(IMAGE_DIR) and
        os.path.exists(TRAIN_CSV_PATH) and
        os.path.exists(TEST_CSV_PATH)
    ),
    reason="Data files not found"
)
@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_dataloader_shapes(batch_size):
    """
    Test whether the DataLoader outputs images with correct shapes under different batch sizes.
    """
    dataloader = get_dataloaders(
        image_dir=IMAGE_DIR,
        csv_path=TRAIN_CSV_PATH,
        batch_size=batch_size
    )
    for images, labels in dataloader:
        # If the current batch is exactly less than batch_size, images.shape[0] will also decrease
        assert (
            images.shape == (batch_size, 1, 224, 224)
            or images.shape[0] < batch_size
        ), (
            f"DataLoader batch image shape {images.shape} "
            f"does not match expected ({batch_size}, 1, 224, 224)"
        )
        assert labels.shape[0] == images.shape[0], (
            f"Label batch size {labels.shape[0]} does not match "
            f"image batch size {images.shape[0]}"
        )
        # Only test the first batch
        break
