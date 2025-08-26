"""
FLOWER Client implementation for ROI dataset FL
"""
import flwr as fl
import torch
from typing import Dict, List, Tuple, Optional
from flwr.common import NDArrays, Scalar
import numpy as np
import os
import sys

# Add flowe directory to Python path
flowe_dir = os.path.dirname(os.path.abspath(__file__))
if flowe_dir not in sys.path:
    sys.path.insert(0, flowe_dir)

from flower_model import FlowerConvolutionalNet, train_model, evaluate_model, calculate_detailed_metrics


class ROIFlowerClient(fl.client.NumPyClient):
    """FLOWER client for ROI classification"""
    
    def __init__(self, 
                 client_id: int,
                 train_loader,
                 val_loader,
                 num_samples: int,
                 device: str = "cuda"):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_samples = num_samples
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.model = FlowerConvolutionalNet()
        
        print(f"FLOWER Client {client_id} initialized with {num_samples} training samples")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters"""
        return self.model.get_weights()
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters"""
        self.model.set_weights(parameters)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model and return updated parameters"""
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Get training config
        epochs = int(config.get("epochs", 1))
        learning_rate = float(config.get("learning_rate", 1e-5))
        
        # Train model
        train_loss, num_samples = train_model(
            self.model, 
            self.train_loader, 
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device
        )
        
        # Return updated parameters and metrics
        metrics = {
            "train_loss": train_loss,
            "client_id": self.client_id,
            "num_samples": num_samples
        }
        
        return self.get_parameters({}), num_samples, metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model and return metrics"""
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Evaluate model
        val_loss, accuracy, num_samples = evaluate_model(
            self.model, 
            self.val_loader, 
            device=self.device
        )
        
        # Calculate detailed metrics
        detailed_metrics = calculate_detailed_metrics(
            self.model, 
            self.val_loader, 
            device=self.device
        )
        
        # Prepare metrics
        metrics = {
            "accuracy": accuracy,
            "f1": detailed_metrics["f1"],
            "precision": detailed_metrics["precision"],
            "recall": detailed_metrics["recall"],
            "client_id": self.client_id,
            "num_samples": num_samples
        }
        
        return val_loss, num_samples, metrics


def create_flower_client(client_id: int, 
                        train_loader, 
                        val_loader, 
                        num_samples: int,
                        device: str = "cuda") -> ROIFlowerClient:
    """Factory function to create FLOWER client"""
    return ROIFlowerClient(client_id, train_loader, val_loader, num_samples, device)


if __name__ == "__main__":
    # Test client creation (requires actual data loaders)
    print("FLOWER Client module loaded successfully")
    print("Use with actual data loaders for testing")
