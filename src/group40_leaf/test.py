from PIL import Image
import os

def inspect_dataset(image_dir):
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            label = int(filename.split('.')[0])  # Assuming the label is part of the filename
            labels.append(label)
    print(f"Minimum Label: {min(labels)}, Maximum Label: {max(labels)}, Total Classes: {len(set(labels))}")

inspect_dataset("/home/justine/MLOps/leaf-classification/images_full")  # Path to the image directory