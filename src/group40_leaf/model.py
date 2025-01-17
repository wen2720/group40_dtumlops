import timm
import torch.nn as nn

# Function to create the model
def convnext(num_classes):
    '''
    Creates a ConvNeXt model with a single-channel input and a specified 
    number of output classes.

            Parameters:
                num_classes (int): The number of output classes.

            Returns:
                nn.Module: The ConvNeXt model.
    '''
    # Load the pretrained ConvNeXt model
    model = timm.create_model('convnext_tiny', pretrained=True)
    # Modify the first convolution layer to accept 1 channel
    model.stem[0] = nn.Conv2d(1, model.stem[0].out_channels, kernel_size=model.stem[0].kernel_size, stride=model.stem[0].stride, padding=model.stem[0].padding, bias=True)
    # Modify the output layer to have the specified number of classes
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model
