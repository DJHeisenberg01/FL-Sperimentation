from dataclasses import dataclass
import numpy as np
import time
import json
from typing import List, Dict, Union

@dataclass
class ClientResources:
    """Classe essenziale per rappresentare diverse risorse computazionali dei client"""
    compute_power: float  # Potere computazionale (1.0 = baseline)
    bandwidth: float      # Larghezza di banda in Mbps
    reliability: float    # Probabilità di completamento riuscito (0-1)
    
    # Profili di qualità predefiniti
    QUALITY_PROFILES = {
        "ottimali": {
            "compute_power": [1.5, 2.0],    # Workstation/Server potenti (GPU dedicate, CPU multi-core)
            "bandwidth": [8.0, 10.0],       # Fibra ottica/5G (connessioni enterprise)
            "reliability": [0.95, 1.0]      # Quasi sempre disponibili (infrastruttura stabile)
        },
        "bilanciati": {
            "compute_power": [0.8, 1.5],    # Desktop standard a workstation (CPU moderne)
            "bandwidth": [4.0, 8.0],        # 4G+ a fibra standard (connessioni domestiche/ufficio)
            "reliability": [0.85, 0.95]     # Abbastanza affidabili (occasionali disconnessioni)
        },
        "scarsi": {
            "compute_power": [0.3, 0.8],    # IoT/dispositivi mobili (smartphone, tablet, edge devices)
            "bandwidth": [1.0, 4.0],        # 3G/WiFi lento (connessioni limitate)
            "reliability": [0.7, 0.85]      # Spesso disconnessi (mobilità, limitazioni energetiche)
        }
    }
    
    def get_client_category(self) -> str:
        """
        Determina la categoria del client basata sui suoi parametri
        
        Returns:
            Stringa che descrive la categoria del client
        """
        # Analizza i parametri per determinare la categoria
        if (self.compute_power >= 1.5 and 
            self.bandwidth >= 8.0 and 
            self.reliability >= 0.95):
            return "OTTIMALE"
        elif (self.compute_power <= 0.8 and 
              self.bandwidth <= 4.0 and 
              self.reliability <= 0.85):
            return "SCARSO"
        else:
            return "BILANCIATO"
    
    def get_detailed_description(self) -> str:
        """
        Restituisce una descrizione dettagliata del client
        
        Returns:
            Stringa con descrizione completa delle risorse
        """
        category = self.get_client_category()
        return (f"[{category}] Power={self.compute_power:.2f}, "
                f"Bandwidth={self.bandwidth:.2f}Mbps, "
                f"Reliability={self.reliability:.2f}")

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
    
    @staticmethod
    def create_from_profile(quality_level: str) -> 'ClientResources':
        """
        Crea ClientResources basato su un profilo di qualità predefinito
        
        Args:
            quality_level: Uno tra 'ottimali', 'bilanciati', 'scarsi'
            
        Returns:
            ClientResources configurato secondo il profilo scelto
        """
        if quality_level not in ClientResources.QUALITY_PROFILES:
            raise ValueError(f"Profilo qualità '{quality_level}' non riconosciuto. "
                           f"Valori disponibili: {list(ClientResources.QUALITY_PROFILES.keys())}")
        
        profile = ClientResources.QUALITY_PROFILES[quality_level]
        
        return ClientResources(
            compute_power=np.random.uniform(*profile["compute_power"]),
            bandwidth=np.random.uniform(*profile["bandwidth"]),
            reliability=np.random.uniform(*profile["reliability"])
        )
    
    @staticmethod
    def generate_clients_from_config(config: Dict) -> List['ClientResources']:
        """
        Genera una lista di client basata sulla configurazione
        
        Args:
            config: Dizionario di configurazione contenente client_configuration
            
        Returns:
            Lista di ClientResources configurati
        """
        client_config = config.get('client_configuration', {})
        num_clients = client_config.get('num_clients', 4)
        quality_level = client_config.get('client_quality', 'bilanciati')
        
        # Genera distribuzione mista se richiesta
        if quality_level == 'misti':
            return ClientResources._generate_mixed_clients(num_clients, client_config)
        
        # Genera client uniformi per il profilo specificato
        clients = []
        for i in range(num_clients):
            client = ClientResources.create_from_profile(quality_level)
            clients.append(client)
        
        return clients
    
    @staticmethod
    def _generate_mixed_clients(num_clients: int, config: Dict) -> List['ClientResources']:
        """
        Genera una distribuzione mista di client con diversi profili di qualità
        """
        # Distribuzione predefinita per profilo misto
        mix_distribution = config.get('mix_distribution', {
            'ottimali': 0.2,    # 20% client ottimali
            'bilanciati': 0.6,  # 60% client bilanciati  
            'scarsi': 0.2       # 20% client scarsi
        })
        
        # Filtra i campi di commento e converti i valori in float
        filtered_distribution = {}
        for key, value in mix_distribution.items():
            if not key.startswith('_') and isinstance(value, (int, float)):
                filtered_distribution[key] = float(value)
        
        # Se non ci sono valori validi, usa la distribuzione di default
        if not filtered_distribution:
            filtered_distribution = {
                'ottimali': 0.2,
                'bilanciati': 0.6,
                'scarsi': 0.2
            }
        
        clients = []
        
        # Aggiungi variabilità al seed per evitare sempre la stessa sequenza
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        for i in range(num_clients):
            # Selezione casuale basata sulla distribuzione
            rand_val = np.random.random()
            cumulative = 0
            
            for quality, probability in filtered_distribution.items():
                cumulative += probability
                if rand_val <= cumulative:
                    client = ClientResources.create_from_profile(quality)
                    clients.append(client)
                    break
        
        return clients
    
    @staticmethod
    def get_quality_ranges(quality_level: str) -> Dict:
        """
        Restituisce i range per un livello di qualità specifico
        
        Args:
            quality_level: Livello di qualità richiesto
            
        Returns:
            Dizionario con i range per compute_power, bandwidth, reliability
        """
        if quality_level not in ClientResources.QUALITY_PROFILES:
            raise ValueError(f"Profilo qualità '{quality_level}' non riconosciuto")
        
        return ClientResources.QUALITY_PROFILES[quality_level].copy()
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Valida la configurazione dei client
        
        Args:
            config: Configurazione da validare
            
        Returns:
            True se la configurazione è valida
        """
        client_config = config.get('client_configuration', {})
        
        # Controlla campi obbligatori
        if 'num_clients' not in client_config:
            print("Warning: 'num_clients' non specificato, usando default (4)")
        
        quality_level = client_config.get('client_quality', 'bilanciati')
        valid_qualities = list(ClientResources.QUALITY_PROFILES.keys()) + ['misti']
        
        if quality_level not in valid_qualities:
            print(f"Error: 'client_quality' deve essere uno tra: {valid_qualities}")
            return False
        
        return True

    @staticmethod
    def update_config_profile(config_path: str, new_profile: str, num_clients: int = None) -> bool:
        """
        Aggiorna il profilo client nel file di configurazione
        
        Args:
            config_path: Percorso al file config.json
            new_profile: Nuovo profilo da impostare
            num_clients: Numero di client (opzionale, mantiene quello esistente se None)
            
        Returns:
            True se l'aggiornamento è andato a buon fine
        """
        try:
            # Carica configurazione esistente
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Assicurati che esista la sezione client_configuration
            if 'client_configuration' not in config:
                config['client_configuration'] = {}
            
            # Aggiorna il profilo
            config['client_configuration']['client_quality'] = new_profile
            
            # Aggiorna numero client se specificato
            if num_clients is not None:
                config['client_configuration']['num_clients'] = num_clients
            
            # Salva configurazione aggiornata
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Configurazione aggiornata: profilo='{new_profile}'" + 
                  (f", num_clients={num_clients}" if num_clients else ""))
            return True
            
        except Exception as e:
            print(f"Errore aggiornamento configurazione: {e}")
            return False

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
        """Genera una configurazione casuale delle risorse per un client (metodo legacy)"""
        return ClientResources(
            compute_power=np.random.uniform(0.5, 2.0),    # Da 0.5x a 2x baseline
            bandwidth=np.random.uniform(1.0, 10.0),       # 1-10 Mbps
            reliability=np.random.uniform(0.8, 1.0)       # 80-100% affidabilità
        )
    
    def generate_clients_from_config(self, config: Dict) -> List[ClientResources]:
        """
        Genera client utilizzando la configurazione del profilo di qualità
        
        Args:
            config: Configurazione completa del sistema
            
        Returns:
            Lista di ClientResources configurati
        """
        if not ClientResources.validate_config(config):
            print("Fallback alla generazione casuale per configurazione non valida")
            client_config = config.get('client_configuration', {})
            num_clients = client_config.get('num_clients', 4)
            return [self.generate_random_resources() for _ in range(num_clients)]
        
        return ClientResources.generate_clients_from_config(config)
    
    def load_and_generate_clients(self, config_path: str) -> List[ClientResources]:
        """
        Carica configurazione da file e genera client
        
        Args:
            config_path: Percorso al file di configurazione JSON
            
        Returns:
            Lista di ClientResources configurati
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return self.generate_clients_from_config(config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Errore caricamento configurazione {config_path}: {e}")
            print("Utilizzando generazione casuale di default")
            return [self.generate_random_resources() for _ in range(4)]
        
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
