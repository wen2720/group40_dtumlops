import pandas as pd
import torch
from data import get_dataloaders
from model import convnext
from loguru import logger
import typer

logger.add("evaluate.log", level="DEBUG", rotation="100 MB")

def evaluate_model(batch_size:int=32, model_path:str="model.pth"):
    logger.info("Starting evaluation")
    logger.info(f"Evaluating model from: {model_path}")
    logger.info(f"Dataset: {csv_path}, Batch size: {batch_size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    image_dir = "data/images"
    csv_path = "data/train.csv"

    # Load data
    dataloader = get_dataloaders(image_dir, csv_path, batch_size, num_images=None)
    logger.info("Dataloader initialized")
    
    # Determine the number of classes
    data = pd.read_csv(csv_path)
    num_classes = data['species'].nunique()
    logger.info(f"Number of classes: {num_classes}")
    
    # Load the model
    model = convnext(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    logger.info("Model loaded successfully")
    model.eval()
    all_predictions = []
    with torch.no_grad():
        logger.info("Starting inference")
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            if batch_idx % 10 == 0:
                logger.debug(f"Processed batch {batch_idx}/{len(dataloader)}")
    logger.info("Inference completed")
    logger.info(f"Total predictions made: {len(all_predictions)}")
    logger.debug(f"Predictions: {all_predictions}")


if __name__ == "__main__":
    logger.add("evaluation.log", level="INFO", rotation="10 MB")
    typer.run(evaluate_model)
