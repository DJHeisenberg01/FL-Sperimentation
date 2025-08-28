"""
Custom Framework Runner - Adapter to run your FL system for comparison
"""
import sys
import os
import time
import subprocess
import threading
import signal
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# Add FLProject to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FLProject'))

try:
    from federated_server import FederatedServer
    from aggregation_policies import UniformAggregation, PowerAwareAggregation
    from start_clients import main as start_clients_main
except ImportError as e:
    print(f"Warning: Could not import custom FL modules: {e}")


class CustomFrameworkRunner:
    """Runs your custom FL framework and collects metrics"""
    
    def __init__(self, 
                 dataset_path: str,
                 num_clients: int = 4,
                 policy: str = "uniform",
                 fl_algorithm: str = "fedavg",
                 num_rounds: int = 2):
        self.dataset_path = dataset_path
        self.num_clients = num_clients
        self.policy = policy
        self.fl_algorithm = fl_algorithm
        self.num_rounds = num_rounds
        self.server_process = None
        self.client_processes = []
        
        # Paths
        self.fl_project_path = os.path.join(os.path.dirname(__file__), '..')
        self.config_path = os.path.join(self.fl_project_path, 'cfg', 'config.json')
        self.temp_config_path = os.path.join(self.fl_project_path, 'cfg', 'temp_test_config.json')
    
    def create_test_config(self) -> str:
        """Create temporary config for testing"""
        # Load base config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Modify for testing
        config.update({
            "global_epoch": self.num_rounds,
            "local_epoch": 1,
            "num_clients": self.num_clients,
            "fl_algorithm": self.fl_algorithm,
            "learning_rate": 1e-5,
            "batch_size": 16,
            "partial_aggregation_enabled": True,
            "min_clients_for_aggregation": min(2, self.num_clients),
            "client_response_timeout": 120  # Longer timeout for testing
        })
        
        # Add client configuration section if not present
        if "client_configuration" not in config:
            config["client_configuration"] = {}
        
        # Update client configuration to use the specified number of clients
        config["client_configuration"].update({
            "num_clients": self.num_clients,
            "client_quality": "bilanciati"  # Default to balanced clients for testing
        })
        
        # Save temporary config
        with open(self.temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return self.temp_config_path
    
    def start_server(self) -> bool:
        """Start the custom FL server"""
        try:
            config_path = self.create_test_config()
            
            # Command to start server
            cmd = [
                sys.executable,
                "start_server.py",
                "--config", config_path,
                "--policy", self.policy
            ]
            
            print(f"Starting custom server with command: {' '.join(cmd)}")
            
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.fl_project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait longer for server to start and check if it's listening
            print("Waiting for server to initialize...")
            for i in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                
                # Check if server process is still running
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    print(f"❌ Server failed to start: {stderr}")
                    return False
                
                # Check if server is listening on port
                if self._check_server_listening():
                    print("✅ Custom server started successfully")
                    return True
            
            print("❌ Server didn't start listening within timeout")
            return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def _check_server_listening(self) -> bool:
        """Check if server is listening on the configured port"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 5000))
            sock.close()
            return result == 0
        except:
            return False
    
    def start_clients(self) -> bool:
        """Start the custom FL clients"""
        try:
            # Command to start clients with the same config as server
            cmd = [
                sys.executable,
                "start_clients.py",
                "--mode", "resource",
                "--policy", self.policy,
                "--num-clients", str(self.num_clients),
                "--config", self.temp_config_path  # Use same config as server
            ]
            
            print(f"Starting custom clients with command: {' '.join(cmd)}")
            
            # Start client process with correct working directory
            client_process = subprocess.Popen(
                cmd,
                cwd=self.fl_project_path,  # Ensure correct working directory
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.client_processes.append(client_process)
            
            # Wait a bit for clients to connect
            time.sleep(15)  # Increased wait time for resource generation and connections
            
            print("Custom clients started")
            return True
            
        except Exception as e:
            print(f"Error starting clients: {e}")
            return False
    
    def wait_for_completion(self, timeout: int = None) -> bool:
        """Wait for training to complete"""
        try:
            if self.server_process:
                print("Waiting for FL training to complete (no timeout)...")
                
                # Check if clients connected by monitoring server output
                start_time = time.time()
                clients_connected = False
                
                while time.time() - start_time < 120:  # Wait up to 120s for connections (increased from 30s)
                    if self.server_process.poll() is not None:
                        # Server terminated early
                        stdout, stderr = self.server_process.communicate()
                        print(f"Server terminated early: {stderr}")
                        return False
                    
                    # Check server logs for client connections
                    if self._check_client_connections():
                        clients_connected = True
                        print("✅ Clients connected successfully")
                        break
                    
                    time.sleep(2)
                
                if not clients_connected:
                    print("❌ No clients connected within timeout")
                    self._print_process_outputs()
                    return False
                
                # Wait for server to complete training (NO TIMEOUT)
                try:
                    print("⏳ Training in progress... (waiting indefinitely)")
                    self.server_process.wait()  # No timeout - wait indefinitely
                    print("✅ Training completed successfully")
                    return True
                except Exception as e:
                    print(f"❌ Error during training: {e}")
                    return False
                
        except Exception as e:
            print(f"❌ Error waiting for completion: {e}")
            self._print_process_outputs()
            return False
    
    def _check_training_completed(self) -> bool:
        """Check if training has completed by examining server logs"""
        try:
            logs_dir = os.path.join(self.fl_project_path, 'logs')
            today = datetime.now().strftime('%d%m')
            
            for subdir in os.listdir(logs_dir):
                if subdir.startswith(today):
                    log_date_dir = os.path.join(logs_dir, subdir, 'FL-Server-LOG')
                    if os.path.exists(log_date_dir):
                        log_files = [f for f in os.listdir(log_date_dir) if f.endswith('.log')]
                        if log_files:
                            latest_log = max(log_files)
                            log_path = os.path.join(log_date_dir, latest_log)
                            
                            # Check for completion indicators
                            with open(log_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                return any(phrase in content for phrase in [
                                    "== done ==",
                                    "Federated training finished",
                                    "get best test loss",
                                    "best model at round"
                                ])
            return False
        except:
            return False
    
    def _check_client_connections(self) -> bool:
        """Check if clients have connected by examining recent server logs"""
        try:
            # Find latest server log
            logs_dir = os.path.join(self.fl_project_path, 'logs')
            today = datetime.now().strftime('%d%m')
            
            for subdir in os.listdir(logs_dir):
                if subdir.startswith(today):
                    log_date_dir = os.path.join(logs_dir, subdir, 'FL-Server-LOG')
                    if os.path.exists(log_date_dir):
                        log_files = [f for f in os.listdir(log_date_dir) if f.endswith('.log')]
                        if log_files:
                            latest_log = max(log_files)
                            log_path = os.path.join(log_date_dir, latest_log)
                            
                            # Check for client connections in log
                            with open(log_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if "The Federated Process is Starting" in content:
                                    return True
                                if "connected" in content and "Client Sid:" in content:
                                    return True
            return False
        except:
            return False
    
    def _print_process_outputs(self):
        """Print outputs from server and client processes for debugging"""
        print("\n🔍 Process Debug Information:")
        
        # Server output - extended debugging
        if self.server_process:
            try:
                # Check if process is still running
                if self.server_process.poll() is None:
                    print("🟢 Server process is still running")
                    # Try to read any buffered output without blocking
                    import select
                    import sys
                    if hasattr(select, 'select'):  # Unix-like systems
                        pass  # For Windows, we'll use a different approach
                else:
                    stdout, stderr = self.server_process.communicate()
                    if stdout:
                        print(f"📄 Server STDOUT:\n{stdout}")
                    if stderr:
                        print(f"⚠️ Server STDERR:\n{stderr}")
            except Exception as e:
                print(f"❌ Could not read server output: {e}")
        
        # Client output - full output
        for i, process in enumerate(self.client_processes):
            try:
                if process.poll() is None:
                    print(f"🟢 Client {i} process is still running")
                    # Force terminate to get output
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                else:
                    stdout, stderr = process.communicate()
                
                if stdout:
                    print(f"📄 Client {i} STDOUT:\n{stdout}")
                if stderr:
                    print(f"⚠️ Client {i} STDERR:\n{stderr}")
            except Exception as e:
                print(f"❌ Could not read client {i} output: {e}")
        
        # Read server log file
        try:
            logs_dir = os.path.join(self.fl_project_path, 'logs')
            today = datetime.now().strftime('%d%m')
            
            server_log_dir = os.path.join(logs_dir, today, 'FL-Server-LOG')
            if os.path.exists(server_log_dir):
                log_files = [f for f in os.listdir(server_log_dir) if f.endswith('.log')]
                if log_files:
                    latest_log = sorted(log_files)[-1]
                    log_path = os.path.join(server_log_dir, latest_log)
                    
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    print(f"\n📋 Server Log Content ({latest_log}):")
                    print("=" * 60)
                    print(log_content)
                    print("=" * 60)
        except Exception as e:
            print(f"❌ Could not read server log: {e}")
    
    def stop_processes(self):
        """Stop all running processes"""
        # Stop client processes
        for process in self.client_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        # Stop server process
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                try:
                    self.server_process.kill()
                except:
                    pass
        
        # Clean up temp config
        if os.path.exists(self.temp_config_path):
            try:
                os.remove(self.temp_config_path)
            except:
                pass
    
    def extract_metrics_from_logs(self) -> Dict[str, float]:
        """Extract final metrics from server logs"""
        try:
            # Find latest server log
            logs_dir = os.path.join(self.fl_project_path, 'logs')
            
            # Get today's date folder
            today = datetime.now().strftime('%d%m')
            log_date_dir = None
            
            for subdir in os.listdir(logs_dir):
                if subdir.startswith(today):
                    log_date_dir = os.path.join(logs_dir, subdir, 'FL-Server-LOG')
                    break
            
            if not log_date_dir or not os.path.exists(log_date_dir):
                print("No server logs found")
                return self._default_metrics()
            
            # Find latest log file
            log_files = [f for f in os.listdir(log_date_dir) if f.endswith('.log')]
            if not log_files:
                print("No log files found")
                return self._default_metrics()
            
            latest_log = max(log_files)
            log_path = os.path.join(log_date_dir, latest_log)
            
            # Parse log file
            metrics = self._parse_log_file(log_path)
            return metrics
            
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            return self._default_metrics()
    
    def _parse_log_file(self, log_path: str) -> Dict[str, float]:
        """Parse log file to extract metrics"""
        metrics = self._default_metrics()
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📋 Parsing log file: {log_path}")
            
            # Look for final metrics - search case insensitive
            lines = content.split('\n')
            for line in reversed(lines):
                line_lower = line.lower()
                
                if "get best test loss" in line_lower:
                    try:
                        # Extract number from line
                        import re
                        numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', line)
                        if numbers:
                            loss_val = float(numbers[-1])
                            if loss_val != float('inf') and not pd.isna(loss_val):
                                metrics["final_loss"] = loss_val
                                print(f"✅ Found final loss: {loss_val}")
                    except Exception as e:
                        print(f"❌ Error parsing loss: {e}")
                        
                elif "get best acc" in line_lower:
                    try:
                        import re
                        numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', line)
                        if numbers:
                            acc_val = float(numbers[-1])
                            if acc_val >= 0 and acc_val <= 1 and not pd.isna(acc_val):
                                metrics["final_accuracy"] = acc_val
                                print(f"✅ Found final accuracy: {acc_val}")
                    except Exception as e:
                        print(f"❌ Error parsing accuracy: {e}")
                        
                elif "get best f1" in line_lower:
                    try:
                        import re
                        numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', line)
                        if numbers:
                            f1_val = float(numbers[-1])
                            if f1_val >= 0 and f1_val <= 1 and not pd.isna(f1_val):
                                metrics["final_f1"] = f1_val
                                print(f"✅ Found final F1: {f1_val}")
                    except Exception as e:
                        print(f"❌ Error parsing F1: {e}")
                        
                elif "get best precision" in line_lower:
                    try:
                        import re
                        numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', line)
                        if numbers:
                            prec_val = float(numbers[-1])
                            if prec_val >= 0 and prec_val <= 1 and not pd.isna(prec_val):
                                metrics["final_precision"] = prec_val
                                print(f"✅ Found final precision: {prec_val}")
                    except Exception as e:
                        print(f"❌ Error parsing precision: {e}")
                        
                elif "get best recall" in line_lower:
                    try:
                        import re
                        numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', line)
                        if numbers:
                            recall_val = float(numbers[-1])
                            if recall_val >= 0 and recall_val <= 1 and not pd.isna(recall_val):
                                metrics["final_recall"] = recall_val
                                print(f"✅ Found final recall: {recall_val}")
                    except Exception as e:
                        print(f"❌ Error parsing recall: {e}")
            
            print(f"📊 Final parsed metrics: {metrics}")
        
        except Exception as e:
            print(f"❌ Error parsing log file {log_path}: {e}")
        
        return metrics
    
    def _default_metrics(self) -> Dict[str, float]:
        """Return default metrics structure"""
        return {
            "final_loss": float('inf'),
            "final_accuracy": 0.0,
            "final_f1": 0.0,
            "final_precision": 0.0,
            "final_recall": 0.0
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete custom FL experiment"""
        print(f"Starting Custom Framework experiment...")
        print(f"Policy: {self.policy}, Algorithm: {self.fl_algorithm}, Clients: {self.num_clients}, Rounds: {self.num_rounds}")
        
        start_time = time.time()
        
        try:
            # Start server
            if not self.start_server():
                raise Exception("Failed to start server")
            
            # Start clients
            if not self.start_clients():
                raise Exception("Failed to start clients")
            
            # Wait for completion (NO TIMEOUT)
            if not self.wait_for_completion():
                raise Exception("Training failed")
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Extract metrics
            metrics = self.extract_metrics_from_logs()
            
            results = {
                "framework": "Custom",
                "strategy": f"{self.policy}_{self.fl_algorithm}",
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "training_time": training_time,
                "success": True,
                **metrics
            }
            
            print(f"✅ Custom framework completed in {training_time:.2f}s")
            return results
            
        except Exception as e:
            print(f"Custom framework experiment failed: {str(e)}")
            return {
                "framework": "Custom",
                "strategy": f"{self.policy}_{self.fl_algorithm}",
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "training_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "final_loss": float('inf'),
                "final_accuracy": 0.0,
                "final_f1": 0.0,
                "final_precision": 0.0,
                "final_recall": 0.0
            }
        
        finally:
            # Always cleanup
            self.stop_processes()


def run_custom_experiment(dataset_path: str,
                         policy: str = "uniform",
                         fl_algorithm: str = "fedavg",
                         num_clients: int = 4,
                         num_rounds: int = 2) -> Dict[str, Any]:
    """
    Run a single custom framework experiment
    """
    runner = CustomFrameworkRunner(
        dataset_path=dataset_path,
        num_clients=num_clients,
        policy=policy,
        fl_algorithm=fl_algorithm,
        num_rounds=num_rounds
    )
    
    return runner.run_experiment()


if __name__ == "__main__":
    # Test custom framework
    dataset_path = "../dataset/Cropped_ROI"
    
    policies = ["uniform", "power"]
    algorithms = ["fedavg", "fedyogi"]
    
    for policy in policies:
        for algorithm in algorithms:
            print(f"\n{'='*50}")
            print(f"Testing Custom Framework: {policy} + {algorithm}")
            print('='*50)
            
            results = run_custom_experiment(
                dataset_path=dataset_path,
                policy=policy,
                fl_algorithm=algorithm,
                num_clients=4,
                num_rounds=2
            )
            
            print(f"Results: {results}")
