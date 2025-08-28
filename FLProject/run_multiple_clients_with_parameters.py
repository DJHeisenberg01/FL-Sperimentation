import os
import time
import json
import threading
import argparse
import numpy as np
from dataset_splitter_clients import DatasetSplitterClients
from federated_client import FederatedClient
from federated_server import FederatedServer
from client_resources import ClientResources
from aggregation_policies import (
    UniformAggregation, PowerAwareAggregation, 
    ReliabilityAwareAggregation, BandwidthAwareAggregation, 
    HybridAggregation
)

# Default paths
IMAGES_PATH = os.path.join('dataset', 'Cropped_ROI')
CLIENTS_PATHS = os.path.join('dataset', 'Clients_Resource_Test')
CONFIG_FILE = os.path.join('cfg', 'config.json')

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def create_client_resources_from_config(config):
    """Create client resources using the new profile-based configuration"""
    try:
        # Use the new profile-based generation
        client_resources = ClientResources.generate_clients_from_config(config)
        return client_resources
    except Exception as e:
        print(f"Warning: Error using profile-based generation: {e}")
        print("Falling back to legacy diverse client generation")
        
        # Fallback to legacy generation
        client_config = config.get('client_configuration', {})
        num_clients = client_config.get('num_clients', config.get('num_clients', 4))
        return create_diverse_client_resources_legacy(num_clients)

def create_diverse_client_resources_legacy(num_clients):
    """Legacy method for creating diverse client resources (kept for backward compatibility)"""
    resources = []
    
    # Create different types of clients
    for i in range(num_clients):
        if i == 0:  # High-end client
            resources.append(ClientResources(
                compute_power=2.0,
                bandwidth=10.0,
                reliability=0.95
            ))
        elif i == 1:  # Low-end client
            resources.append(ClientResources(
                compute_power=0.5,
                bandwidth=2.0,
                reliability=0.85
            ))
        elif i == 2:  # Unreliable client
            resources.append(ClientResources(
                compute_power=1.5,
                bandwidth=8.0,
                reliability=0.7
            ))
        else:  # Random client
            resources.append(ClientResources(
                compute_power=np.random.uniform(0.8, 1.8),
                bandwidth=np.random.uniform(3.0, 9.0),
                reliability=np.random.uniform(0.8, 0.95)
            ))
    
    return resources

def start_client_with_resources(config_file, dataset_path, client_resources):
    """Start a federated client with specific resources"""
    FederatedClient(config_file, dataset_path, client_resources)

def start_server_with_policy(config_file, aggregation_policy):
    """Start the federated server with a specific aggregation policy"""
    server = FederatedServer(config_file, aggregation_policy)
    server.run()

def run_experiment(policy_name, aggregation_policy, config_file=CONFIG_FILE, 
                  splitting_dir=CLIENTS_PATHS, images_path=IMAGES_PATH):
    """Run a complete federated learning experiment with specific policy"""
    
    print(f"\n{'='*50}")
    print(f"STARTING EXPERIMENT: {policy_name}")
    print(f"{'='*50}")
    
    # Ensure paths are absolute
    config_file = os.path.abspath(config_file)
    splitting_dir = os.path.abspath(splitting_dir)
    images_path = os.path.abspath(images_path)
    
    # Load configuration
    try:
        config = load_json(config_file)
        num_clients = config['num_clients']
        print(f"Configuration loaded from {config_file}")
        print(f"Using FL algorithm: {config.get('fl_algorithm', 'fedavg')}")
        print(f"Aggregation policy: {policy_name}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Create experiment-specific clients directory
    experiment_dir = os.path.join(splitting_dir, policy_name.lower().replace(' ', '_'))
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Split Dataset
    try:
        splitter = DatasetSplitterClients(experiment_dir, images_path, num_clients)
        print(f"Splitting dataset into {num_clients} groups")
        print(f"Source: {images_path}")
        print(f"Destination: {experiment_dir}")
        splitter.split()
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return

    # Create client resources using new profile-based configuration
    client_resources = create_client_resources_from_config(config)
    
    # Enhanced logging with client profiles
    client_config = config.get('client_configuration', {})
    client_quality = client_config.get('client_quality', 'Not specified')
    print(f"\n🤖 CLIENT PROFILE CONFIGURATION:")
    print(f"   Profile: {client_quality}")
    print(f"   Number of clients: {len(client_resources)}")
    
    print(f"\nClient Resource Configuration:")
    for i, resources in enumerate(client_resources):
        print(f"  Client {i}: {resources.get_detailed_description()}")

    # Start server in separate thread
    print(f"\nStarting server with {policy_name} aggregation policy...")
    server_thread = threading.Thread(
        target=start_server_with_policy,
        args=(config_file, aggregation_policy),
        name=f"Server_{policy_name}"
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to initialize
    time.sleep(10)

    # Start clients
    print(f"\nStarting {num_clients} clients...")
    client_threads = []
    for i in range(num_clients):
        dataset_path = os.path.join(experiment_dir, f'client_{i}')
        if not os.path.exists(dataset_path):
            print(f"Warning: Client dataset path does not exist: {dataset_path}")
            continue
            
        thread = threading.Thread(
            target=start_client_with_resources,
            args=(config_file, dataset_path, client_resources[i]),
            name=f"Client_{i}_{policy_name}"
        )
        thread.daemon = True
        thread.start()
        print(f"Started client {i} with dataset: {dataset_path}")
        print(f"  Resources: Power={client_resources[i].compute_power:.2f}, "
              f"Bandwidth={client_resources[i].bandwidth:.2f}Mbps, "
              f"Reliability={client_resources[i].reliability:.2f}")
        
        # Wait between client starts to avoid overwhelming the server
        time.sleep(5)  # Increased delay between client starts in resource testing
        client_threads.append(thread)

    # Wait for experiment to complete
    print(f"\nExperiment '{policy_name}' running...")
    print("Waiting for clients to complete training...")
    
    # Wait for a reasonable amount of time (adjust based on your global_epoch setting)
    experiment_timeout = 300  # 5 minutes timeout
    start_time = time.time()
    
    while time.time() - start_time < experiment_timeout:
        alive_threads = [t for t in client_threads if t.is_alive()]
        if not alive_threads:
            break
        time.sleep(10)
        print(f"  {len(alive_threads)} clients still running...")
    
    print(f"Experiment '{policy_name}' completed!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Run federated learning experiments with different aggregation policies')
    parser.add_argument('--config', default=CONFIG_FILE, help='Path to config file')
    parser.add_argument('--clients-dir', default=CLIENTS_PATHS, help='Path to clients directory')
    parser.add_argument('--images-dir', default=IMAGES_PATH, help='Path to source images directory')
    parser.add_argument('--policy', choices=['uniform', 'power', 'reliability', 'bandwidth', 'hybrid', 'all'], 
                       default='all', help='Aggregation policy to test')
    parser.add_argument('--delay', type=int, default=60, help='Delay between experiments in seconds')
    
    args = parser.parse_args()
    
    # Define experiments
    experiments = {
        'uniform': ('Uniform Aggregation (FedAvg)', UniformAggregation()),
        'power': ('Power-Aware Aggregation', PowerAwareAggregation()),
        'reliability': ('Reliability-Aware Aggregation', ReliabilityAwareAggregation()),
        'bandwidth': ('Bandwidth-Aware Aggregation', BandwidthAwareAggregation(min_bandwidth=3.0)),
        'hybrid': ('Hybrid Aggregation', HybridAggregation())
    }
    
    if args.policy == 'all':
        # Run all experiments
        for exp_key, (exp_name, exp_policy) in experiments.items():
            try:
                run_experiment(exp_name, exp_policy, args.config, args.clients_dir, args.images_dir)
                
                if exp_key != list(experiments.keys())[-1]:  # Not the last experiment
                    print(f"\nWaiting {args.delay} seconds before next experiment...")
                    time.sleep(args.delay)
                    
            except KeyboardInterrupt:
                print(f"\nExperiment interrupted by user")
                break
            except Exception as e:
                print(f"Error in experiment '{exp_name}': {e}")
                continue
    else:
        # Run single experiment
        if args.policy in experiments:
            exp_name, exp_policy = experiments[args.policy]
            run_experiment(exp_name, exp_policy, args.config, args.clients_dir, args.images_dir)
        else:
            print(f"Unknown policy: {args.policy}")
            return
    
    print(f"\n{'='*50}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*50}")
    print("Check the logs and CSV output files for detailed results.")
    print("Each experiment used different aggregation policies and client resource configurations.")

if __name__ == '__main__':
    main()
