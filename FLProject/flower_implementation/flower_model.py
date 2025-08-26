"""
Model definition for FLOWER FL framework - matches your original system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
import sys
import os

# Add flowe directory to Python path for Ray workers
flowe_dir = os.path.dirname(os.path.abspath(__file__))
if flowe_dir not in sys.path:
    sys.path.insert(0, flowe_dir)


class FlowerConvolutionalNet(nn.Module):
    """
    FLOWER version of your ConvolutionalNet using ResNet18
    """
    
    def __init__(self, num_classes: int = 2):
        super(FlowerConvolutionalNet, self).__init__()
        
        # Use ResNet18 backbone (same as your system)
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights as numpy arrays (FLOWER format)"""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model weights from numpy arrays (FLOWER format)"""
        params_dict = zip(self.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)


def create_model() -> FlowerConvolutionalNet:
    """Factory function to create model instance"""
    return FlowerConvolutionalNet(num_classes=2)


def train_model(model: FlowerConvolutionalNet, 
                train_loader, 
                epochs: int = 1,
                learning_rate: float = 1e-5,
                device: str = "cuda") -> Tuple[float, int]:
    """
    Train model for specified epochs
    Returns: (loss, num_samples)
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            epoch_samples += data.size(0)
        
        total_loss += epoch_loss
        total_samples += epoch_samples
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, total_samples


def evaluate_model(model: FlowerConvolutionalNet, 
                  val_loader, 
                  device: str = "cuda") -> Tuple[float, float, int]:
    """
    Evaluate model on validation set
    Returns: (loss, accuracy, num_samples)
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, total_samples


def calculate_detailed_metrics(model: FlowerConvolutionalNet, 
                             val_loader, 
                             device: str = "cuda") -> Dict[str, float]:
    """
    Calculate detailed metrics (F1, Precision, Recall) like your system
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    try:
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)
    except:
        f1 = precision = recall = accuracy = 0.0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    # Test model creation and basic operations
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test weight extraction and setting
    weights = model.get_weights()
    print(f"Extracted {len(weights)} weight arrays")
    
    # Test with dummy data
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
