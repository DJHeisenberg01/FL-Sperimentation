"""
Dataset handler for FLOWER FL framework using the same ROI dataset
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys

# Add flowe directory to Python path for Ray workers
flowe_dir = os.path.dirname(os.path.abspath(__file__))
if flowe_dir not in sys.path:
    sys.path.insert(0, flowe_dir)


class ROIDataset(Dataset):
    """Custom dataset for ROI images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class FlowerDatasetSplitter:
    """Dataset splitter compatible with FLOWER framework"""
    
    def __init__(self, dataset_path: str, num_clients: int = 4):
        self.dataset_path = dataset_path
        self.num_clients = num_clients
        self.roi_csv_path = os.path.join(dataset_path, 'roi_annotation.csv')
        
        # Load CSV data
        if not os.path.exists(self.roi_csv_path):
            raise FileNotFoundError(f"ROI annotation file not found: {self.roi_csv_path}")
        
        self.df = pd.read_csv(self.roi_csv_path)
        
        # Define transforms (same as your original system)
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_full_image_path(self, filename: str, class_label: int) -> str:
        """Get full path to image file"""
        class_dir = "damaged" if class_label == 0 else "healthy"
        return os.path.join(self.dataset_path, class_dir, filename)
    
    def split_for_flower(self) -> List[Tuple[DataLoader, DataLoader, int]]:
        """
        Split dataset for FLOWER clients
        Returns list of (train_loader, val_loader, dataset_size) for each client
        """
        # Use StratifiedKFold for consistent splitting with your system
        skf = StratifiedKFold(n_splits=self.num_clients, random_state=42, shuffle=True)
        client_data = []
        
        for i, (_, client_indices) in enumerate(skf.split(self.df['filename'], self.df['class'])):
            # Get client subset
            client_df = self.df.iloc[client_indices]
            
            # Further split into train/val (90/10 like your system)
            train_indices, val_indices = train_test_split(
                client_df.index, 
                test_size=0.1, 
                random_state=42, 
                stratify=client_df['class']
            )
            
            # Prepare train data
            train_paths = []
            train_labels = []
            for idx in train_indices:
                row = self.df.iloc[idx]
                img_path = self.get_full_image_path(row['filename'], row['class'])
                if os.path.exists(img_path):
                    train_paths.append(img_path)
                    train_labels.append(row['class'])
            
            # Prepare validation data
            val_paths = []
            val_labels = []
            for idx in val_indices:
                row = self.df.iloc[idx]
                img_path = self.get_full_image_path(row['filename'], row['class'])
                if os.path.exists(img_path):
                    val_paths.append(img_path)
                    val_labels.append(row['class'])
            
            # Create datasets
            train_dataset = ROIDataset(train_paths, train_labels, self.transform_train)
            val_dataset = ROIDataset(val_paths, val_labels, self.transform_val)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=16,  # Same as your system
                shuffle=True, 
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=16, 
                shuffle=False, 
                num_workers=2
            )
            
            client_data.append((train_loader, val_loader, len(train_dataset)))
            
            print(f"Client {i}: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return client_data
    
    def get_test_loader(self) -> DataLoader:
        """Create a test loader using a subset of data for global evaluation"""
        # Use 20% of data for testing
        test_indices = self.df.sample(frac=0.2, random_state=42).index
        
        test_paths = []
        test_labels = []
        for idx in test_indices:
            row = self.df.iloc[idx]
            img_path = self.get_full_image_path(row['filename'], row['class'])
            if os.path.exists(img_path):
                test_paths.append(img_path)
                test_labels.append(row['class'])
        
        test_dataset = ROIDataset(test_paths, test_labels, self.transform_val)
        return DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)


def load_flower_data(dataset_path: str, num_clients: int = 4) -> Tuple[List[Tuple[DataLoader, DataLoader, int]], DataLoader]:
    """
    Main function to load data for FLOWER FL
    Returns: (client_data_list, test_loader)
    """
    splitter = FlowerDatasetSplitter(dataset_path, num_clients)
    client_data = splitter.split_for_flower()
    test_loader = splitter.get_test_loader()
    
    return client_data, test_loader


if __name__ == "__main__":
    # Test the dataset splitting
    dataset_path = "../dataset/Cropped_ROI"
    client_data, test_loader = load_flower_data(dataset_path, num_clients=4)
    
    print(f"Created data for {len(client_data)} clients")
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Test loading a batch
    for i, (train_loader, val_loader, size) in enumerate(client_data[:1]):
        print(f"\nClient {i} - Dataset size: {size}")
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch[0].shape}, Labels: {batch[1]}")
        break
