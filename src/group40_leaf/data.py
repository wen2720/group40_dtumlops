import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger

logger.add("data.log", level="DEBUG", rotation="100 MB")

class LeafDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None, num_images=None):
        logger.info(f"Initializing LeafDataset with image_dir={image_dir} and csv_path={csv_path}")

        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(self.data)} rows")
        self.transform = transform
        
        # Create a mapping from species to class indices
        self.species_to_label = {species: idx for idx, species in enumerate(self.data['species'].unique())}
        self.data['label'] = self.data['species'].map(self.species_to_label)

        # Limit the number of images if num_images is specified
        if num_images is not None:
            self.data = self.data.head(num_images)

        # Create a mapping from species to class indices
        self.species_to_label = {species: idx for idx, species in enumerate(self.data['species'].unique())}
        self.data['label'] = self.data['species'].map(self.species_to_label)
        logger.info(f"Generated label mapping for {len(self.species_to_label)} unique species")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['id']
        label = row['label']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('L')  # Convert to grayscale (1 channel)
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(image_dir, csv_path, batch_size=32, num_images=100):
    logger.info(f"Creating dataloader for image_dir={image_dir}, csv_path={csv_path}, batch_size={batch_size}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])
    logger.info("Transformations: Resize to (224, 224), ToTensor, Normalize(mean=0.5, std=0.5)")
    dataset = LeafDataset(image_dir=image_dir, csv_path=csv_path, transform=transform, num_images=num_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataloader created with {len(dataloader)} batches")

    return dataloader
