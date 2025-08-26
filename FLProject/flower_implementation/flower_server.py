"""
FLOWER Server implementation with custom strategies for comparison
"""
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Metrics, NDArrays, Parameters
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from collections import OrderedDict
import torch
import os
import sys

# Add flowe directory to Python path
flowe_dir = os.path.dirname(os.path.abspath(__file__))
if flowe_dir not in sys.path:
    sys.path.insert(0, flowe_dir)

from flower_model import FlowerConvolutionalNet, evaluate_model


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with detailed logging"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics = []
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate training results with logging"""
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log round metrics
        if aggregated_metrics:
            round_data = {
                "round": server_round,
                "num_clients": len(results),
                "strategy": "FedAvg",
                "train_loss": aggregated_metrics.get("train_loss", 0.0)
            }
            self.round_metrics.append(round_data)
            print(f"Round {server_round} FedAvg - Clients: {len(results)}, Loss: {aggregated_metrics.get('train_loss', 0.0):.4f}")
        
        return aggregated_weights, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results with logging"""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if aggregated_metrics:
            print(f"Round {server_round} FedAvg Eval - Loss: {aggregated_loss:.4f}, "
                  f"Acc: {aggregated_metrics.get('accuracy', 0.0):.4f}, "
                  f"F1: {aggregated_metrics.get('f1', 0.0):.4f}")
        
        return aggregated_loss, aggregated_metrics


class CustomFedProx(FedProx):
    """Custom FedProx strategy with detailed logging"""
    
    def __init__(self, proximal_mu: float = 0.01, **kwargs):
        super().__init__(proximal_mu=proximal_mu, **kwargs)
        self.round_metrics = []
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate training results with logging"""
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log round metrics
        if aggregated_metrics:
            round_data = {
                "round": server_round,
                "num_clients": len(results),
                "strategy": "FedProx",
                "train_loss": aggregated_metrics.get("train_loss", 0.0)
            }
            self.round_metrics.append(round_data)
            print(f"Round {server_round} FedProx - Clients: {len(results)}, Loss: {aggregated_metrics.get('train_loss', 0.0):.4f}")
        
        return aggregated_weights, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results with logging"""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if aggregated_metrics:
            print(f"Round {server_round} FedProx Eval - Loss: {aggregated_loss:.4f}, "
                  f"Acc: {aggregated_metrics.get('accuracy', 0.0):.4f}, "
                  f"F1: {aggregated_metrics.get('f1', 0.0):.4f}")
        
        return aggregated_loss, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average function for metrics aggregation"""
    if not metrics:
        return {}
    
    # Calculate weighted averages
    total_samples = sum(num_examples for num_examples, _ in metrics)
    
    if total_samples == 0:
        return {}
    
    weighted_metrics = {}
    
    # Get all metric keys from first entry
    if metrics[0][1]:
        for key in metrics[0][1].keys():
            if key in ['client_id']:  # Skip non-aggregatable metrics
                continue
                
            weighted_sum = sum(
                num_examples * m.get(key, 0.0) 
                for num_examples, m in metrics 
                if key in m
            )
            weighted_metrics[key] = weighted_sum / total_samples
    
    return weighted_metrics


def create_flower_strategy(strategy_name: str, 
                          min_fit_clients: int = 2,
                          min_available_clients: int = 4,
                          fraction_fit: float = 1.0,
                          fraction_evaluate: float = 1.0) -> Union[CustomFedAvg, CustomFedProx]:
    """
    Factory function to create FLOWER strategies
    """
    base_params = {
        "min_fit_clients": min_fit_clients,
        "min_available_clients": min_available_clients,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "fit_metrics_aggregation_fn": weighted_average,
    }
    
    if strategy_name.lower() == "fedavg":
        return CustomFedAvg(**base_params)
    elif strategy_name.lower() == "fedprox":
        return CustomFedProx(proximal_mu=0.01, **base_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def get_initial_parameters() -> Parameters:
    """Get initial model parameters for FLOWER server"""
    model = FlowerConvolutionalNet()
    weights = model.get_weights()
    return fl.common.ndarrays_to_parameters(weights)


class FlowerServerConfig:
    """Configuration for FLOWER server"""
    
    def __init__(self,
                 strategy_name: str = "fedavg",
                 num_rounds: int = 2,
                 min_fit_clients: int = 2,
                 min_available_clients: int = 4,
                 fraction_fit: float = 1.0):
        self.strategy_name = strategy_name
        self.num_rounds = num_rounds
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        
        # Create strategy
        self.strategy = create_flower_strategy(
            strategy_name=strategy_name,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit
        )
    
    def get_config(self) -> Dict:
        """Get configuration for client fit/evaluate"""
        return {
            "epochs": 1,
            "learning_rate": 1e-5,
            "strategy": self.strategy_name
        }


if __name__ == "__main__":
    # Test strategy creation
    strategies = ["fedavg", "fedprox"]
    
    for strategy_name in strategies:
        strategy = create_flower_strategy(strategy_name)
        print(f"{strategy_name} strategy created: {type(strategy).__name__}")
    
    # Test initial parameters
    initial_params = get_initial_parameters()
    print(f"Initial parameters created with {len(fl.common.parameters_to_ndarrays(initial_params))} arrays")
