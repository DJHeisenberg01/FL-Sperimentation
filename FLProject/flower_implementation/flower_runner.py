"""
FLOWER FL Training Runner - Simulates federated learning with FLOWER framework
"""
import flwr as fl
from flwr.simulation import start_simulation
import torch
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import time
import os
import sys
from datetime import datetime

# Add flower_implementation directory to Python path for imports
flower_impl_dir = os.path.dirname(os.path.abspath(__file__))
if flower_impl_dir not in sys.path:
    sys.path.insert(0, flower_impl_dir)

from flower_dataset import load_flower_data
from flower_client import create_flower_client
from flower_server import FlowerServerConfig, get_initial_parameters


class FlowerSimulationRunner:
    """Runs FLOWER FL simulation and collects metrics"""
    
    def __init__(self, 
                 dataset_path: str,
                 num_clients: int = 4,
                 strategy: str = "fedavg",
                 num_rounds: int = 2):
        self.dataset_path = dataset_path
        self.num_clients = num_clients
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.results = []
        
        # Load dataset
        print(f"Loading FLOWER dataset for {num_clients} clients...")
        self.client_data, self.test_loader = load_flower_data(dataset_path, num_clients)
        
        # Setup server config
        self.server_config = FlowerServerConfig(
            strategy_name=strategy,
            num_rounds=num_rounds,
            min_fit_clients=min(2, num_clients),
            min_available_clients=num_clients,
            fraction_fit=1.0
        )
    
    def client_fn(self, cid: str):
        """Create client function for simulation"""
        client_id = int(cid)
        train_loader, val_loader, num_samples = self.client_data[client_id]
        
        return create_flower_client(
            client_id=client_id,
            train_loader=train_loader,
            val_loader=val_loader,
            num_samples=num_samples
        )
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run FLOWER simulation and return results"""
        print(f"Starting FLOWER simulation with {self.strategy.upper()} strategy...")
        
        start_time = time.time()
        
        try:
            # Run simulation
            hist = start_simulation(
                client_fn=self.client_fn,
                num_clients=self.num_clients,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=self.server_config.strategy,
                client_resources={
                    "num_cpus": 1,
                    "num_gpus": 0.0 if not torch.cuda.is_available() else 0.25
                }
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Extract final metrics
            final_metrics = self._extract_final_metrics(hist)
            
            results = {
                "framework": "FLOWER",
                "strategy": self.strategy,
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "training_time": training_time,
                "success": True,
                **final_metrics
            }
            
            print(f"FLOWER simulation completed in {training_time:.2f}s")
            return results
            
        except Exception as e:
            print(f"FLOWER simulation failed: {str(e)}")
            return {
                "framework": "FLOWER",
                "strategy": self.strategy,
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "training_time": 0.0,
                "success": False,
                "error": str(e),
                "final_loss": float('inf'),
                "final_accuracy": 0.0,
                "final_f1": 0.0,
                "final_precision": 0.0,
                "final_recall": 0.0
            }
    
    def _extract_final_metrics(self, hist) -> Dict[str, float]:
        """Extract final metrics from simulation history"""
        try:
            print(f"📊 Extracting metrics from FLOWER history...")
            print(f"History losses_distributed: {hist.losses_distributed}")
            print(f"History metrics_distributed: {hist.metrics_distributed}")
            
            # Initialize default metrics
            final_metrics = {
                "final_loss": float('inf'),
                "final_accuracy": 0.0,
                "final_f1": 0.0,
                "final_precision": 0.0,
                "final_recall": 0.0,
            }
            
            # Extract final loss
            if hist.losses_distributed and len(hist.losses_distributed) > 0:
                final_metrics["final_loss"] = hist.losses_distributed[-1][1]
                print(f"✅ Final loss: {final_metrics['final_loss']}")
            
            # Extract other metrics from the last round of evaluation
            if hist.metrics_distributed and len(hist.metrics_distributed) > 0:
                # hist.metrics_distributed is a dict with metric names as keys
                # Each value is a list of (round_num, value) tuples
                print(f"📋 Available metrics: {list(hist.metrics_distributed.keys())}")
                
                for metric_name, metric_values in hist.metrics_distributed.items():
                    if metric_values and len(metric_values) > 0:
                        # Get the last value for this metric
                        latest_value = metric_values[-1][1]  # (round_num, value) -> value
                        
                        if metric_name == "accuracy":
                            final_metrics["final_accuracy"] = latest_value
                        elif metric_name == "f1":
                            final_metrics["final_f1"] = latest_value
                        elif metric_name == "precision":
                            final_metrics["final_precision"] = latest_value
                        elif metric_name == "recall":
                            final_metrics["final_recall"] = latest_value
                
                print(f"✅ Final accuracy: {final_metrics['final_accuracy']}")
                print(f"✅ Final F1: {final_metrics['final_f1']}")
                print(f"✅ Final precision: {final_metrics['final_precision']}")
                print(f"✅ Final recall: {final_metrics['final_recall']}")
            
            return final_metrics
            
        except Exception as e:
            print(f"❌ Error extracting metrics: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return {
                "final_loss": float('inf'),
                "final_accuracy": 0.0,
                "final_f1": 0.0,
                "final_precision": 0.0,
                "final_recall": 0.0,
            }


def run_flower_experiment(dataset_path: str,
                         strategy: str = "fedavg",
                         num_clients: int = 4,
                         num_rounds: int = 2) -> Dict[str, Any]:
    """
    Run a single FLOWER experiment
    """
    runner = FlowerSimulationRunner(
        dataset_path=dataset_path,
        num_clients=num_clients,
        strategy=strategy,
        num_rounds=num_rounds
    )
    
    return runner.run_simulation()


if __name__ == "__main__":
    # Test FLOWER simulation
    dataset_path = "../dataset/Cropped_ROI"
    
    strategies = ["fedavg", "fedprox"]
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing FLOWER with {strategy.upper()}")
        print('='*50)
        
        results = run_flower_experiment(
            dataset_path=dataset_path,
            strategy=strategy,
            num_clients=4,
            num_rounds=2
        )
        
        print(f"Results: {results}")
