from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple

class AggregationPolicy(ABC):
    """Classe base per le politiche di aggregazione"""
    @abstractmethod
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        """Seleziona i client per la partecipazione all'attuale round"""
        pass
        
    @abstractmethod
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        """Calcola i pesi di aggregazione per i client selezionati"""
        pass

class UniformAggregation(AggregationPolicy):
    """Simple uniform aggregation (FedAvg)"""
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        return available_clients
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        weight = 1.0 / len(selected_clients)
        return {client: weight for client in selected_clients}

class PowerAwareAggregation(AggregationPolicy):
    """Politica di aggregazione che favorisce i client con maggiore potere computazionale"""
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        # Filtra i client in base alla soglia minima di potere computazionale
        min_power = 0.5  # Minimo 50% del potere di base
        return [
            client for client in available_clients
            if client_resources[client].compute_power >= min_power
        ]
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        total_power = sum(client_resources[c].compute_power for c in selected_clients)
        return {
            client: client_resources[client].compute_power / total_power
            for client in selected_clients
        }

class ReliabilityAwareAggregation(AggregationPolicy):
    """Politica di aggregazione che considera l'affidabilità dei client"""
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        # Filtra i client in base alla soglia minima di affidabilità
        min_reliability = 0.8  # Minimo 80% di affidabilità
        return [
            client for client in available_clients
            if client_resources[client].reliability >= min_reliability
        ]
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        total_reliability = sum(client_resources[c].reliability for c in selected_clients)
        return {
            client: client_resources[client].reliability / total_reliability
            for client in selected_clients
        }

class BandwidthAwareAggregation(AggregationPolicy):
    """Politica di aggregazione che considera le condizioni di rete"""
    def __init__(self, min_bandwidth: float = 2.0):
        self.min_bandwidth = min_bandwidth
        
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        return [
            client for client in available_clients
            if client_resources[client].bandwidth >= self.min_bandwidth
        ]
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        total_bandwidth = sum(client_resources[c].bandwidth for c in selected_clients)
        return {
            client: client_resources[client].bandwidth / total_bandwidth
            for client in selected_clients
        }

class HybridAggregation(AggregationPolicy):
    """Combina più metriche per la selezione e la pesatura dei client"""
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        selected = []
        for client in available_clients:
            resources = client_resources[client]
            # Calcola il punteggio composito
            score = (resources.compute_power * 0.4 +
                    resources.reliability * 0.4 +
                    (resources.bandwidth / 10.0) * 0.2)  # bandwidth normalizzata
            if score >= 0.6:  # Soglia minima
                selected.append(client)
        return selected
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        scores = {}
        for client in selected_clients:
            resources = client_resources[client]
            scores[client] = (resources.compute_power * 0.4 +
                            resources.reliability * 0.4 +
                            (resources.bandwidth / 10.0) * 0.2)
        
        total_score = sum(scores.values())
        return {client: score / total_score for client, score in scores.items()}
