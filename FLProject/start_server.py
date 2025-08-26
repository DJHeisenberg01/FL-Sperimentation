#!/usr/bin/env python3
"""
Standalone Federated Learning Server
Run this script to start only the federated server
"""
import sys
import os
import argparse

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated_server import FederatedServer

def main():
    parser = argparse.ArgumentParser(description='Run Federated Learning Server')
    parser.add_argument('--config', default='cfg/config.json', 
                       help='Path to configuration file')
    parser.add_argument('--policy', choices=['uniform', 'power', 'reliability', 'bandwidth', 'hybrid'], 
                       default=None, help='Aggregation policy to use (optional)')
    
    args = parser.parse_args()
    
    # Initialize aggregation policy if specified
    aggregation_policy = None
    if args.policy:
        from aggregation_policies import (
            UniformAggregation, PowerAwareAggregation, 
            ReliabilityAwareAggregation, BandwidthAwareAggregation, 
            HybridAggregation
        )
        
        policies = {
            'uniform': UniformAggregation(),
            'power': PowerAwareAggregation(),
            'reliability': ReliabilityAwareAggregation(),
            'bandwidth': BandwidthAwareAggregation(min_bandwidth=3.0),
            'hybrid': HybridAggregation()
        }
        
        aggregation_policy = policies.get(args.policy)
        print(f"Using {args.policy} aggregation policy")
    else:
        print("Using default (uniform) aggregation policy")
    
    # Start server
    print(f"Starting Federated Learning Server...")
    print(f"Configuration: {args.config}")
    print("Server will listen for client connections...")
    print("Press Ctrl+C to stop the server")
    
    try:
        server = FederatedServer(args.config, aggregation_policy)
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
