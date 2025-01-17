import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import convnext
from torch.profiler import profile, ProfilerActivity
from loguru import logger
import wandb
import typer
import pickle

logger.add("train.log", level="DEBUG", rotation="100 MB")

def save_model_pickle(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def train_model(epochs:int=10, batch_size:int=32, lr:float=1e-3):
    run = wandb.init(
        project="group40_leaf",
        config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs},
    )
    logger.info("Starting the training process")
    logger.info(f"Configuration: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    image_dir = "data/images"
    csv_path = "data/train.csv"

    # Get the dataloader
    dataloader = get_dataloaders(image_dir, csv_path, batch_size)
    logger.info("Dataloader initialized")

    # Determine the number of classes
    data = pd.read_csv(csv_path)
    num_classes = data['species'].nunique()
    logger.info(f"Number of classes: {num_classes}")

#    with profile(
#        activities=[ProfilerActivity.CPU],  # Use only CPU activities
#        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
#        on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler"),
#        with_stack=True,
#    ) as prof:
        # Initialize model, loss, and optimizer
    model = convnext(num_classes=num_classes).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1) 
            correct += (preds == labels).sum().item() 
            total += labels.size(0) 
            
        train_loss = running_loss / len(dataloader) 
        train_accuracy = correct / total

        wandb.log({ "epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy, "learning_rate": optimizer.param_groups[0]['lr'] })

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}")
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {running_loss / len(dataloader):.4f}")
    
    # Save the trained model
    with open("models/leaf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    artifact = wandb.Artifact(
        name="group40_leaf",
        type="model",
        description="A model trained to classify leaf images",
        metadata={"train_loss": train_loss, "train_accuracy": train_accuracy},
    )
    artifact.add_file("models/leaf_model.pkl")
    run.log_artifact(artifact)
#    wandb.save("leaf_model.pth")

#    # Export profiling trace
#    prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    logger.add("training.log", level="INFO", rotation="10 MB")
    typer.run(train_model)
