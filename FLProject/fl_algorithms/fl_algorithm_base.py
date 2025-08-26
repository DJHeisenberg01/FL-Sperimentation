from abc import ABC, abstractmethod
import numpy as np

class FLAlgorithm(ABC):
    def __init__(self, config, initial_weights=None):
        self.config = config
        self.current_weights = initial_weights
        
    @abstractmethod
    def aggregate_weights(self, client_weights, client_sizes=None):
        """
        Aggregate weights from multiple clients
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes (optional)
        Returns:
            Aggregated weights
        """
        pass
        
    def update_current_weights(self, weights):
        """
        Update the current weights of the global model
        Args:
            weights: New weights to set
        """
        self.current_weights = weights
