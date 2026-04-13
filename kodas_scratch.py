"""
By: Beatričė Urbaitė
LSP: 2425051
Variant: part-time studies, 1st variant
File 2: code for model made from scratch
"""
# LIBRARIES
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import os

# CONFIGURATION
# Select GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 5
LOAD_PATH = "scratch_model_weights.pth" # Path to load existing weights
MODEL_SAVE_PATH = "fine_tuned_50ep.pth" # Path to save final weights

# VOC IDs to local IDs (0-4) mapping
CLASS_MAP = {0: 0, 8: 1, 12: 2, 3: 3, 13: 4}

# ARCHITECTURE
class UNetScratch(nn.Module):
    def __init__(self, num_classes):
        super(UNetScratch, self).__init__()

        # Helper for repetitive double convolution blocks
        def double_conv(input_channels, output_channels):
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels), # Stabilizes training
                nn.ReLU(inplace=True), # Activation function
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder layers (contracting path)
        self.enc1 = double_conv(3, 64)
        self.enc2 = double_conv(64, 128)
        self.pool = nn.MaxPool2d(2) # Downsampling

        # Latent space (bottom of the U)
        self.bottleneck = double_conv(128, 256)

        # Decoder layers (expansive path)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # Upsampling
        self.dec2 = double_conv(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(64, 64)

        # Final layer to output num_classes scores per pixel
        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass through encoder, bottleneck, and decoder
        s1 = self.enc1(x); p1 = self.pool(s1)
        s2 = self.enc2(p1); p2 = self.pool(s2)
        b = self.bottleneck(p2)
        d2 = self.upconv2(b); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = self.dec1(d1)
        return {'out': self.final_layer(d1)}

# DATA PREPARATION
# Image pipeline: Resize, convert to Tensor, and Normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mask pipeline: Resize using NEAREST to keep class labels as integers
mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
])

class FilteredVOC(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train"):
        # Load VOC 2012 segmentation dataset
        self.dataset = VOCSegmentation(root=root, year="2012", image_set=image_set, download=True)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img_t = transform(img) # Apply image transforms
        mask_np = np.array(mask_transform(mask)) # Apply mask resize

        new_mask = np.zeros_like(mask_np) # Create empty background mask
        # Map specific VOC classes to our new 0-4 labels
        for voc_idx, our_idx in CLASS_MAP.items():
            new_mask[mask_np == voc_idx] = our_idx
        return img_t, torch.tensor(new_mask, dtype=torch.long)

# TRAINING LOOP
if __name__ == "__main__":
    # Create DataLoader for batch processing and shuffling
    train_loader = DataLoader(FilteredVOC(root="./data"), batch_size=BATCH_SIZE, shuffle=True)
    model = UNetScratch(NUM_CLASSES).to(DEVICE)

    # Resume training if weights file exists
    if os.path.exists(LOAD_PATH):
        print(f"Resuming from {LOAD_PATH}...")
        model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))

    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Apply higher weights (3.0) to animal classes to combat class imbalance
    class_weights = torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"50 epoch training happening {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()        # Clear previous gradients
            output = model(imgs)['out']   # Get predictions
            loss = criterion(output, masks) # Calculate loss
            loss.backward()              # Backpropagation
            optimizer.step()             # Update weights

            running_loss += loss.item()

        # Log average loss per epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{50}] | Avg Loss: {avg_loss:.4f}")

    # Finalize and save trained model
    print(f"\nTraining is finished. Saved in directory: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)