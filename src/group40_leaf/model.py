import timm
import torch.nn as nn

# Function to create the model
def convnext(num_classes):
    model = timm.create_model('convnext_tiny', pretrained=True)
    # Modify the first convolution layer to accept 1 channel
    model.stem[0] = nn.Conv2d(1, model.stem[0].out_channels, kernel_size=model.stem[0].kernel_size, stride=model.stem[0].stride, padding=model.stem[0].padding, bias=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
