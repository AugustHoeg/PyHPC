import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from multistep_model import MultiStepSegmentationModel
import torch.nn.functional as F
from tqdm import tqdm

class LiverDataset(Dataset):
    def __init__(self, csv_path, image_type='T1', target_shape=(64, 128, 128)):
        """
        Args:
            csv_path: Path to the CSV file with image paths
            image_type: 'T1' or 'T2'
            target_shape: Target shape for interpolation (D, H, W)
        """
        self.df = pd.read_csv(csv_path)
        self.image_type = image_type
        self.target_shape = target_shape
        
        # Filter out rows where the specified image type is not available
        img_col = f'{image_type}_img'
        mask_col = f'{image_type}_mask'
        self.df = self.df[self.df[img_col].notna() & self.df[mask_col].notna()]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image and mask
        img_path = row[f'{self.image_type}_img']
        mask_path = row[f'{self.image_type}_mask']
        
        # Load using nibabel
        img_nib = nib.load(img_path)
        mask_nib = nib.load(mask_path)
        
        # Convert to numpy arrays
        img = img_nib.get_fdata()
        mask = mask_nib.get_fdata()
        
        # Normalize image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())
        
        # Add channel dimension
        img = torch.from_numpy(img).float().unsqueeze(0)  # [1, D, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1, D, H, W]
        
        # Interpolate to target shape
        img = F.interpolate(img.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        
        return img, mask

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = LiverDataset(args.cirrhotic_csv, image_type=args.image_type)
    if args.verbose:
        print(f"Training dataset size: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    model = MultiStepSegmentationModel(
        image_channels=1,
        initial_mask_channels=1,
        num_classes=1,
        base_cnn_hidden_features=args.cnn_features,
        patch_size_d=16,  # D/4
        patch_size_h=32,  # H/4
        patch_size_w=32,  # W/4
        verbose=args.verbose
    ).to(device)
    
    if args.verbose:
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits_coarse, logits_refine1, logits_final = model(images, masks)
            
            # Calculate loss (using final logits)
            loss = criterion(logits_final, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if args.verbose and batch_idx % 10 == 0:
                print(f"\nBatch {batch_idx}:")
                print(f"  Image shape: {images.shape}")
                print(f"  Mask shape: {masks.shape}")
                print(f"  Coarse logits shape: {logits_coarse.shape}")
                print(f"  Refine1 logits shape: {logits_refine1.shape}")
                print(f"  Final logits shape: {logits_final.shape}")
                print(f"  Loss: {loss.item():.4f}")
        
        # Print epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train liver segmentation model')
    
    # Dataset parameters
    parser.add_argument('--cirrhotic-csv', type=str, default='cirrhotic_subjects_data.csv',
                      help='Path to cirrhotic subjects CSV')
    parser.add_argument('--image-type', type=str, choices=['T1', 'T2'], default='T1',
                      help='Type of image to use (T1 or T2)')
    
    # Model parameters
    parser.add_argument('--cnn-features', type=int, default=4,
                      help='Number of features in CNN blocks')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--save-every', type=int, default=10,
                      help='Save checkpoint every N epochs')
    
    # Other parameters
    parser.add_argument('--verbose', action='store_true',default=False,
                      help='Print detailed information during training')
    
    args = parser.parse_args()
    
    train(args) 