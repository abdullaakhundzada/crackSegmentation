import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from utils import resize_tensor_transform

class CrackDataset(Dataset):
    def __init__(self, data_dir, transform_shape : tuple[int, int]):
        self.data_dir = data_dir
        self.transform = resize_tensor_transform(transform_shape)
        self.images = []
        self.masks = []
        
        # Load images and masks
        image_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")
        
        for img_name in os.listdir(image_dir):
            if img_name.endswith((".jpg", ".png")):
                self.images.append(os.path.join(image_dir, img_name))
                self.masks.append(os.path.join(mask_dir, img_name))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = self.transform(image)
        mask = transforms.ToTensor()(mask)
        
        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Ensure same size by resizing targets if needed
        if predictions.shape != targets.shape:
            targets = nn.functional.interpolate(
                targets.unsqueeze(1) if targets.dim() == 3 else targets,
                size=predictions.shape[2:],
                mode="nearest"
            )
        
        # Flattening
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def forward(self, predictions, targets):
        # Ensure same size for BCE loss as well
        if predictions.shape != targets.shape:
            targets = nn.functional.interpolate(
                targets.unsqueeze(1) if targets.dim() == 3 else targets,
                size=predictions.shape[2:],
                mode="nearest"
            )
        
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce

def train_model(model, train_loader, val_loader, device, 
                dice_loss_weight=0.5, bce_loss_weight=0.5,
                num_epochs=10, save_all=False, checkpoint_dir="", 
                result_path="results.csv", visualize=True):

    results = {"epoch": [], "train_loss" : [], "val_loss" : []}
    criterion = CombinedLoss(dice_weight=dice_loss_weight, bce_weight=bce_loss_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    model.to(device)
    best_val_loss = float("inf")
    
    for epoch in tqdm(range(num_epochs), maxinterval=num_epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        scheduler.step(avg_val_loss)
        
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}")
        tqdm.write(f"Training Loss: {avg_train_loss:.4f}")
        tqdm.write(f"Validation Loss: {avg_val_loss:.4f}")

        results["epoch"].append(epoch)
        results["train_loss"].append(avg_train_loss)
        results["val_loss"].append(avg_val_loss)

        
        # Save best model
        if avg_val_loss < best_val_loss or save_all:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pth')
        
    result_df = pd.DataFrame(results)
    result_df.to_csv(result_path)

    if visualize:
        sns.lineplot(data=result_df, x="epoch", y="val_loss")
        plt.grid()
        plt.show()