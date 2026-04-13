"""
By: Beatričė Urbaitė
LSP: 2425051
Variant: part-time studies, 1st variant
File 1: code for pretrained model
"""

# LIBRARIES
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms, models
import os


# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # using cpu
BATCH_SIZE = 4 # 4 images at a time
EPOCHS = 5 # 5 epochs (full passes thorugh training data) of training
NUM_CLASSES = 5 # using 5 classes
MODEL_SAVE_PATH = "pretrained_model.pth" # a saved file of model weights of not fine-tuned model

# Mapping - VOC labels to my subset
# 0: Background, 8: Cat, 12: Dog, 3: Bird, 13: Horse
CLASS_MAP = {0: 0, 8: 1, 12: 2, 3: 3, 13: 4}

# DATA PREPARATION
image_transform = transforms.Compose([
    transforms.Resize((256, 256)), # resizing the training image to 256x256 pixels
    transforms.ToTensor(), #conversion of jpeg to tensor (matrix of numbers), scale pixel values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # scaled pixels adapted to ImageNet average colours
])

mask_transform = transforms.Compose([
    # Resize mask to match image size (256x256)
    transforms.Resize((256, 256),
    # NEAREST class IDs not blend into fake numbers
    interpolation=transforms.InterpolationMode.NEAREST),
])


class FilteredVOC(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train"):
        # Download and/or initialize Pascal VOC 2012 dataset
        self.dataset = VOCSegmentation(root=root, year="2012", image_set=image_set, download=True)

    def __len__(self):
        # Total amount of images available in the dataset split
        return len(self.dataset)

    def __getitem__(self, index):
        # Take raw image and corresponding ground truth mask by index
        image, mask = self.dataset[index]
        # Image scaling/normalising, spatial resizing to mask
        image_t = image_transform(image)
        mask_np = np.array(mask_transform(mask))

        # Create an empty black mask (all zeros/background) of the same size
        new_mask = np.zeros_like(mask_np)

        # Re-map original VOC class IDs to our specific 0-4 subset
        for voc_idx, our_idx in CLASS_MAP.items():
            # If a pixel matches a VOC animal ID, assign it our new class ID
            new_mask[mask_np == voc_idx] = our_idx

        # Return the processed image and the re-mapped mask as a LongTensor
        return image_t, torch.tensor(new_mask, dtype=torch.long)


# MODEL INITIALIZATION
def get_model(num_classes):
    # Load model with pre-trained ResNet50 features (Transfer Learning)
    model = models.segmentation.deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.DEFAULT')
    # Replace final 21-class layer with a new layer for our 5 classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    # Move model to CPU
    return model.to(DEVICE)


# TRAINING EXECUTION
if __name__ == "__main__":
    # Create the data loader to feed images in batches and shuffle them every epoch
    train_loader = DataLoader(FilteredVOC(root="./data", image_set="train"), batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the modified DeepLabV3 model
    model = get_model(NUM_CLASSES)
    # Use Adam optimizer with a learning rate of 0.0001 for stable weight updates
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Standard loss function for multi-class segmentation tasks
    criterion = nn.CrossEntropyLoss()

    print(f"Training started on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train() # Set model to training mode (enables BatchNorm/Dropout)
        epoch_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            # Move data to the same hardware as the model (GPU or CPU)
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()        # Reset gradients from previous step
            output = model(images)['out'] # Forward pass: get pixel predictions
            loss = criterion(output, masks) # Calculate error against ground truth
            loss.backward()              # Backward pass: compute gradients
            optimizer.step()             # Update model weights

            epoch_loss += loss.item()    # Accumulate loss for tracking

            # Print progress every 20 steps
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] | Step [{i + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Calculate and print average loss for the entire epoch
        print(f"Epoch {epoch + 1} Avg Loss: {epoch_loss / len(train_loader):.4f}")

    # SAVE THE MODEL WEIGHTS
    print(f"\nTraining Complete. Saving model weights to: {MODEL_SAVE_PATH}")
    # Export the learned weights (state_dict) to a file for later testing
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Weight file saved.")