import io
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

app = FastAPI(title="Leaf Classification Inference")

# Load Model from File
model_path = "models/leaf_model.pkl"

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define input transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction Endpoint

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Leaf model inference API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint that takes in an image and returns the predicted class index."""
    # Read file bytes
    image_bytes = await file.read()
    # Convert bytes to a PIL Image
    img = Image.open(io.BytesIO(image_bytes))

    # Ensure image is 'L' (8-bit grayscale)
    if img.mode != "L":
        img = img.convert("L")

    # Apply transform pipeline
    img_t = transform(img)
    # Add batch dimension and move to device
    img_t = img_t.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted_idx = torch.max(outputs, 1)

    # Convert to a normal Python int
    predicted_idx = predicted_idx.item()
    return {"prediction": predicted_idx}
