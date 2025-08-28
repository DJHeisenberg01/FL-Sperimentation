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
        min_power = 0.3  # Abbassato per includere client "scarsi" (era 0.5)
        
        selected_clients = []
        filtered_clients = []
        
        for client in available_clients:
            client_power = client_resources[client].compute_power
            if client_power >= min_power:
                selected_clients.append(client)
            else:
                filtered_clients.append((client, client_power))
        
        # Log della selezione/filtro
        import logging
        logger = logging.getLogger("Federated-Server")
        
        if selected_clients:
            logger.info(f"[POWER POLICY] Selected {len(selected_clients)}/{len(available_clients)} clients above power threshold {min_power}")
            for client in selected_clients:
                power = client_resources[client].compute_power
                logger.info(f"[POWER POLICY] [+] Client {client}: Power={power:.2f} (>= {min_power})")
        
        if filtered_clients:
            logger.info(f"[POWER POLICY] Filtered out {len(filtered_clients)} clients below power threshold {min_power}")
            for client, power in filtered_clients:
                logger.info(f"[POWER POLICY] [-] Client {client}: Power={power:.2f} (< {min_power})")
        
        return selected_clients
        
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
        
        selected_clients = []
        filtered_clients = []
        
        for client in available_clients:
            client_reliability = client_resources[client].reliability
            if client_reliability >= min_reliability:
                selected_clients.append(client)
            else:
                filtered_clients.append((client, client_reliability))
        
        # Log della selezione/filtro
        import logging
        logger = logging.getLogger("Federated-Server")
        
        if selected_clients:
            logger.info(f"[RELIABILITY POLICY] Selected {len(selected_clients)}/{len(available_clients)} clients above reliability threshold {min_reliability}")
            for client in selected_clients:
                reliability = client_resources[client].reliability
                logger.info(f"[RELIABILITY POLICY] [+] Client {client}: Reliability={reliability:.2f} (>= {min_reliability})")
        
        if filtered_clients:
            logger.info(f"[RELIABILITY POLICY] Filtered out {len(filtered_clients)} clients below reliability threshold {min_reliability}")
            for client, reliability in filtered_clients:
                logger.info(f"[RELIABILITY POLICY] [-] Client {client}: Reliability={reliability:.2f} (< {min_reliability})")
        
        return selected_clients
        
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
        selected_clients = []
        filtered_clients = []
        
        for client in available_clients:
            client_bandwidth = client_resources[client].bandwidth
            if client_bandwidth >= self.min_bandwidth:
                selected_clients.append(client)
            else:
                filtered_clients.append((client, client_bandwidth))
        
        # Log della selezione/filtro
        import logging
        logger = logging.getLogger("Federated-Server")
        
        if selected_clients:
            logger.info(f"[BANDWIDTH POLICY] Selected {len(selected_clients)}/{len(available_clients)} clients above bandwidth threshold {self.min_bandwidth}")
            for client in selected_clients:
                bandwidth = client_resources[client].bandwidth
                logger.info(f"[BANDWIDTH POLICY] [+] Client {client}: Bandwidth={bandwidth:.2f}Mbps (>= {self.min_bandwidth})")
        
        if filtered_clients:
            logger.info(f"[BANDWIDTH POLICY] Filtered out {len(filtered_clients)} clients below bandwidth threshold {self.min_bandwidth}")
            for client, bandwidth in filtered_clients:
                logger.info(f"[BANDWIDTH POLICY] [-] Client {client}: Bandwidth={bandwidth:.2f}Mbps (< {self.min_bandwidth})")
        
        return selected_clients
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        total_bandwidth = sum(client_resources[c].bandwidth for c in selected_clients)
        return {
            client: client_resources[client].bandwidth / total_bandwidth
            for client in selected_clients
        }

class HybridAggregation(AggregationPolicy):
    """Combina più metriche per la selezione e la pesatura dei client"""
    def select_clients(self, available_clients: List[str], client_resources: Dict) -> List[str]:
        selected_clients = []
        filtered_clients = []
        min_score = 0.6  # Soglia minima
        
        for client in available_clients:
            resources = client_resources[client]
            # Calcola il punteggio composito
            score = (resources.compute_power * 0.4 +
                    resources.reliability * 0.4 +
                    (resources.bandwidth / 10.0) * 0.2)  # bandwidth normalizzata
            
            if score >= min_score:
                selected_clients.append((client, score))
            else:
                filtered_clients.append((client, score, resources))
        
        # Log della selezione/filtro
        import logging
        logger = logging.getLogger("Federated-Server")
        
        if selected_clients:
            logger.info(f"[HYBRID POLICY] Selected {len(selected_clients)}/{len(available_clients)} clients above composite score threshold {min_score}")
            for client, score in selected_clients:
                resources = client_resources[client]
                logger.info(f"[HYBRID POLICY] [+] Client {client}: Score={score:.3f} (P:{resources.compute_power:.2f}, R:{resources.reliability:.2f}, B:{resources.bandwidth:.1f})")
        
        if filtered_clients:
            logger.info(f"[HYBRID POLICY] Filtered out {len(filtered_clients)} clients below composite score threshold {min_score}")
            for client, score, resources in filtered_clients:
                logger.info(f"[HYBRID POLICY] [-] Client {client}: Score={score:.3f} (P:{resources.compute_power:.2f}, R:{resources.reliability:.2f}, B:{resources.bandwidth:.1f})")
        
        return [client for client, score in selected_clients]
        
    def compute_weights(self, selected_clients: List[str], client_resources: Dict) -> Dict[str, float]:
        scores = {}
        for client in selected_clients:
            resources = client_resources[client]
            scores[client] = (resources.compute_power * 0.4 +
                            resources.reliability * 0.4 +
                            (resources.bandwidth / 10.0) * 0.2)
        
        total_score = sum(scores.values())
        return {client: score / total_score for client, score in scores.items()}
