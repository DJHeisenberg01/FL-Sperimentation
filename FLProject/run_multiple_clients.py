import os
import time
import json
import threading
import argparse
from dataset_splitter_clients import DatasetSplitterClients
from federated_client import FederatedClient

# Default paths
IMAGES_PATH = os.path.join('dataset', 'Cropped_ROI')
CLIENTS_PATHS = os.path.join('dataset', 'Clients_1')
CONFIG_FILE = os.path.join('cfg', 'config.json')


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def start_client(config_file, dataset_path):
    # Backward compatibility: create client without resources
    FederatedClient(config_file, dataset_path)


def main(config_file=CONFIG_FILE, splitting_dir=CLIENTS_PATHS, images_path=IMAGES_PATH):
    # Ensure paths are absolute
    config_file = os.path.abspath(config_file)
    splitting_dir = os.path.abspath(splitting_dir)
    images_path = os.path.abspath(images_path)
    
    # Load configuration
    try:
        config = load_json(config_file)
        
        # Get number of clients from client_configuration if available, otherwise from legacy field
        client_config = config.get('client_configuration', {})
        if 'num_clients' in client_config:
            num_clients = client_config['num_clients']
            print(f"Using num_clients from client_configuration: {num_clients}")
        else:
            num_clients = config.get('num_clients', 4)
            print(f"Using num_clients from legacy configuration: {num_clients}")
            
        print(f"Configuration loaded from {config_file}")
        print(f"Using FL algorithm: {config.get('fl_algorithm', 'fedavg')}")
        
        # Show client profile if configured
        client_quality = client_config.get('client_quality')
        if client_quality:
            print(f"Client quality profile: {client_quality}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Create clients directory if it doesn't exist
    os.makedirs(splitting_dir, exist_ok=True)
    
    # Split Dataset and Erase old files
    try:
        splitter = DatasetSplitterClients(splitting_dir, images_path, num_clients)
        print(f"Splitting dataset into {num_clients} groups")
        print(f"Source: {images_path}")
        print(f"Destination: {splitting_dir}")
        splitter.split()
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return

    # Start clients
    threads = []
    for i in range(num_clients):
        dataset_path = os.path.join(splitting_dir, f'client_{i}')
        if not os.path.exists(dataset_path):
            print(f"Warning: Client dataset path does not exist: {dataset_path}")
            continue
            
        thread = threading.Thread(
            target=start_client, 
            args=(config_file, dataset_path),
            name=f"Client_{i}"
        )
        thread.start()
        print(f"Started client {i} with dataset: {dataset_path}")
        # Wait between client starts to avoid overwhelming the server
        time.sleep(8)  # Increased delay between client starts
        threads.append(thread)

    # Wait for all clients to finish
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple federated learning clients')
    parser.add_argument('--config', default=CONFIG_FILE, help='Path to config file')
    parser.add_argument('--clients-dir', default=CLIENTS_PATHS, help='Path to clients directory')
    parser.add_argument('--images-dir', default=IMAGES_PATH, help='Path to source images directory')
    
    args = parser.parse_args()
    main(args.config, args.clients_dir, args.images_dir)
