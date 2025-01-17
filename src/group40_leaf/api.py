

import io
import pickle

import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import convnext

def convnext(num_classes):
    # Must match the definition you used when training!
    model = timm.create_model('convnext_tiny', pretrained=False)
    # Adjust the stem to accept 1-channel input
    model.stem[0] = nn.Conv2d(
        in_channels=1,
        out_channels=model.stem[0].out_channels,
        kernel_size=model.stem[0].kernel_size,
        stride=model.stem[0].stride,
        padding=model.stem[0].padding,
        bias=True
    )

    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model



app = FastAPI(title="Leaf Classification Inference")



model = convnext(num_classes=99)

########################
# Move model to device
########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

########################
# Define input transforms
########################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


########################
# Prediction Endpoint
########################
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint that takes in an image and returns the predicted class index."""
    # Read file bytes
    image_bytes = await file.read()
    # Convert bytes to a PIL Image
    img = Image.open(io.BytesIO(image_bytes))

    #  ensure image is 'L' (8-bit grayscale).
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
