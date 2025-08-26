from dataclasses import dataclass
import numpy as np
import time

@dataclass
class ClientResources:
    """Classe essenziale per rappresentare diverse risorse computazionali dei client"""
    compute_power: float  # Potere computazionale (1.0 = baseline)
    bandwidth: float      # Larghezza di banda in Mbps
    reliability: float    # Probabilità di completamento riuscito (0-1)
    
    def simulate_computation_time(self, baseline_time: float) -> float:
        """Simulate computation time based on compute power"""
        # variazione casuale (±10%)
        variation = np.random.uniform(0.9, 1.1)
        return (baseline_time / self.compute_power) * variation
    
    def simulate_transmission_time(self, data_size_mb: float) -> float:
        """Simula il tempo di trasmissione basato sulla larghezza di banda"""
        # Aggiunge un po' di jitter di rete (±20%)
        jitter = np.random.uniform(0.8, 1.2)
        return (data_size_mb / self.bandwidth) * jitter
    
    def check_availability(self) -> bool:
        """Controlla se il client è disponibile in base all'affidabilità"""
        return np.random.random() < self.reliability

class ResourceManager:
    """Manager per le risorse dei client"""
    def __init__(self):
        self.client_resources = {}
        self.baseline_epoch_time = None
        
    def add_client(self, client_id: str, resources: ClientResources):
        self.client_resources[client_id] = resources
    
    def get_client_resources(self, client_id: str) -> ClientResources:
        return self.client_resources.get(client_id)
    
    def generate_random_resources(self) -> ClientResources:
        """Genera una configurazione casuale delle risorse per un client"""
        return ClientResources(
            compute_power=np.random.uniform(0.5, 2.0),    # Da 0.5x a 2x baseline
            bandwidth=np.random.uniform(1.0, 10.0),       # 1-10 Mbps
            reliability=np.random.uniform(0.8, 1.0)       # 80-100% affidabilità
        )
        
    def estimate_completion_time(self, client_id: str, data_size_mb: float) -> float:
        """Stima il tempo di completamento per un client"""
        resources = self.get_client_resources(client_id)
        if not resources:
            return 0.0
            
        compute_time = resources.simulate_computation_time(self.baseline_epoch_time or 1.0)
        transmission_time = resources.simulate_transmission_time(data_size_mb)
        return compute_time + transmission_time
    
    def is_client_available(self, client_id: str) -> bool:
        """Controlla se il client è disponibile per il training"""
        resources = self.get_client_resources(client_id)
        return resources and resources.check_availability()

class ResourceAwareTraining:
    """Utility per la stima delle risorse nei client in training"""
    def __init__(self):
        self.baseline_epoch_time = None
        self.resource_manager = ResourceManager()
        
    def estimate_completion_time(self, client_id: str, data_size_mb: float) -> float:
        """Stima il tempo di completamento per un client"""
        resources = self.resource_manager.get_client_resources(client_id)
        if not resources:
            return 0.0
            
        compute_time = resources.simulate_computation_time(self.baseline_epoch_time)
        transmission_time = resources.simulate_transmission_time(data_size_mb)
        return compute_time + transmission_time
    
    def is_client_available(self, client_id: str) -> bool:
        """Controlla se il client è disponibile per il training"""
        resources = self.resource_manager.get_client_resources(client_id)
        return resources and resources.check_availability()
