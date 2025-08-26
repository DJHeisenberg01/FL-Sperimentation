#!/usr/bin/env python3
"""
Standalone Federated Learning Clients
Run this script to start only the federated clients (server must be running separately)
"""
import sys
import os
import argparse
import threading
import time

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated_client import FederatedClient
from dataset_splitter_clients import DatasetSplitterClients
from client_resources import ClientResources
import numpy as np
import json

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def start_single_client(client_path, config_file, client_resources=None):
    """Start a single federated client"""
    try:
        print(f"Starting client with data path: {client_path}")
        client = FederatedClient(config_file, client_path, client_resources)
    except Exception as e:
        print(f"Error starting client {client_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run ONLY Federated Learning Clients (Server must be running separately)')
    parser.add_argument('--config', default='cfg/config.json', 
                       help='Path to configuration file')
    parser.add_argument('--clients-dir', default='dataset/Clients_1',
                       help='Path to clients directory')
    parser.add_argument('--images-dir', default='dataset/Cropped_ROI',
                       help='Path to source images directory')
    parser.add_argument('--mode', choices=['standard', 'resource'], default='standard',
                       help='Client mode: standard (traditional) or resource (with resource simulation)')
    parser.add_argument('--policy', choices=['uniform', 'power', 'reliability', 'bandwidth', 'hybrid'], 
                       default='uniform', help='Aggregation policy (only for resource mode)')
    parser.add_argument('--num-clients', type=int, default=4,
                       help='Number of clients to start')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("FEDERATED LEARNING CLIENTS ONLY")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Configuration: {args.config}")
    print(f"Number of clients: {args.num_clients}")
    print()
    print("WARNING: IMPORTANT: Make sure the server is ALREADY RUNNING!")
    print("   Start server with: python start_server.py")
    print()
    print("Press Ctrl+C to stop all clients")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_json(args.config)
        
        # Prepare dataset splitting
        if args.mode == 'resource':
            policy_name = f"{args.policy}_aggregation"
            clients_dir = os.path.join(args.clients_dir, policy_name)
        else:
            clients_dir = args.clients_dir
        
        # Check if clients directory exists, if not split dataset
        if not os.path.exists(clients_dir) or len(os.listdir(clients_dir)) < args.num_clients:
            print(f"Splitting dataset into {args.num_clients} groups")
            print(f"Source: {args.images_dir}")
            print(f"Destination: {clients_dir}")
            
            splitter = DatasetSplitterClients(clients_dir, args.images_dir, args.num_clients)
            splitter.split()
        else:
            print(f"Using existing client data in: {clients_dir}")
        
        # Create client resources if in resource mode
        client_resources_list = []
        if args.mode == 'resource':
            print("\nGenerating client resources:")
            for i in range(args.num_clients):
                resources = ClientResources(
                    compute_power=np.random.uniform(0.5, 2.0),
                    bandwidth=np.random.uniform(1.0, 10.0),
                    reliability=np.random.uniform(0.8, 1.0)
                )
                client_resources_list.append(resources)
                print(f"  Client {i}: Power={resources.compute_power:.2f}, "
                      f"Bandwidth={resources.bandwidth:.2f}, Reliability={resources.reliability:.2f}")
        else:
            client_resources_list = [None] * args.num_clients
        
        # Start clients in separate threads
        print(f"\nStarting {args.num_clients} federated clients...")
        threads = []
        
        for i in range(args.num_clients):
            client_path = os.path.join(clients_dir, f"client_{i}")
            
            thread = threading.Thread(
                target=start_single_client,
                args=(client_path, args.config, client_resources_list[i])
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
            # Small delay between client starts
            time.sleep(2)
        
        print(f"All {args.num_clients} clients started!")
        print("Clients are now connecting to the server...")
        print("Press Ctrl+C to stop all clients")
        
        # Wait for all threads
        for thread in threads:
            thread.join()
            
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Clients stopped by user")
        print("=" * 50)
    except Exception as e:
        print(f"\nERROR Client startup error: {e}")
        print("Make sure:")
        print("1. The server is running (python start_server.py)")
        print("2. The configuration file exists")
        print("3. The dataset directory exists")
        sys.exit(1)

if __name__ == '__main__':
    main()
