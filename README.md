# Federated Learning Framework Plug-and-Play for Damaged Signal Detection

### System Overview

The system implements an advanced Federated Learning framework that enables distributed training of deep learning models across multiple clients while keeping data locally on each device. The system is designed to simulate realistic scenarios with heterogeneous clients that have different computational, network, and reliability capabilities.

### Key Features

The framework is characterized by the following innovations:

1. **Resource-Aware Simulation**: Each virtual client has different performance characteristics that realistically influence the training behavior  
2. **Adaptive Aggregation Policies**: Implementation of various selection and aggregation strategies based on available resources  
3. **Partial Aggregation**: Ability to proceed with aggregation even when not all clients respond  
4. **Framework Comparison**: Automatic testing system that compares the custom framework with FLOWER  
5. **Multiple FL Algorithms**: Support for FedAvg, FedProx, FedYogi, and FedAdam  

---

## System Architecture

### Main Components

**Federated Server (federated_server.py)**  
The federated server acts as the central coordinator and manages:

- Management of connections with multiple simultaneous clients  
- Maintenance of each client’s resource state  
- Coordination of global training rounds  
- Intelligent client selection based on configurable policies  
- Model aggregation using advanced algorithms  
- Detailed monitoring and logging of performance  

**Federated Client (federated_client.py)**  
Each federated client maintains:

- Local private dataset with autonomous data management  
- Local training capabilities with specific FL algorithms  
- Realistic simulation of its computational resources  
- Asynchronous communication with the server  
- Calculation and reporting of local metrics  

**Resource System (client_resources.py)**  
Handles the simulation of client characteristics:

- Variable computational power (0.5x to 2.0x compared to baseline)  
- Simulated bandwidth (1.0 to 10.0 Mbps)  
- Client reliability (success probability 0.7–1.0)  
- Realistic computation and transmission times  

### Communication Architecture

```
[Server FL] ←→ [Client 1] [Client 2] [Client 3] [Client 4]
     ↑
[Aggregator] + [Policy Selection] + [FL Algorithm]
```

The system uses asynchronous bidirectional communication based on WebSocket, allowing:

- Persistent connections between server and clients  
- Asynchronous sending of models and metrics  
- Robust handling of disconnections and timeouts  
- Universal distribution of updated models  

---

## Client Resource Simulation

### ClientResources Class

The resource simulation is implemented through the `ClientResources` class that models three fundamental aspects:

**Compute Power**  
Represents the client’s computing capacity relative to a baseline device:

- 0.5: IoT devices with very limited hardware (2x slower training)  
- 0.8: Smartphones/Tablets with moderate capacity (25% slower training)  
- 1.0: Standard desktop used as baseline  
- 1.5: High-performance workstation (50% faster training)  
- 2.0: Dedicated Server/GPU (2x faster training)  

**Bandwidth**  
Client’s network connection speed in Megabits per second:

- 1.0–2.0 Mbps: Slow connections (2G/3G, weak WiFi)  
- 3.0–5.0 Mbps: Moderate connections (4G, home WiFi)  
- 6.0–8.0 Mbps: Fast connections (4G+, standard fiber)  
- 9.0–10.0 Mbps: Optimal connections (5G, high-speed fiber)  

**Reliability**  
Probability that the client successfully completes a training round:

- 0.7–0.8: Unstable clients with frequent disconnections  
- 0.85–0.9: Moderately reliable clients  
- 0.95–1.0: Highly stable and reliable clients  

### Realistic Simulation

The system introduces stochastic variability to simulate real conditions:

```
Computation_Time = (Base_Time / Compute_Power) * Random_Variation(0.9, 1.1)
Transmission_Time = (Data_Size_MB / Bandwidth) * Network_Jitter(0.8, 1.2)
```

This implementation allows testing system behavior under variable network conditions and heterogeneous client performance.  

---

## Aggregation Policies

### Implemented Policies

**UniformAggregation (Classic FedAvg)**  
Implements standard federated aggregation where all clients contribute equally:

- Uniform weight for all participants  
- Equivalent to the original FedAvg  
- Suitable for homogeneous client scenarios  

**PowerAwareAggregation**  
Favors clients with greater computational power:

- Weights proportional to computing capacity  
- Improves convergence in heterogeneous scenarios  
- Reduces the impact of slow clients  

**ReliabilityAwareAggregation**  
Considers historical client reliability:

- Higher weight for more stable clients  
- Reduces impact of intermittent clients  
- Improves robustness of training  

**BandwidthAwareAggregation**  
Optimizes for network conditions:

- Considers available bandwidth  
- Filters out clients with inadequate connections  
- Minimum threshold configurable (default 3.0 Mbps)  

**HybridAggregation**  
Combines multiple metrics for optimal selection:

- Composite algorithm balancing power, reliability, and bandwidth  
- Adaptive selection based on combined score  
- Maximum flexibility for complex scenarios  

### Partial Aggregation

The system supports aggregation even when not all clients respond:

- Minimum threshold configurable via `min_clients_for_aggregation`  
- Universal distribution of updated model to all registered clients  
- Automatic timeout handling to avoid indefinite blocking  
- Fallback to alternative strategies when too few clients respond  

---

## Federated Learning Algorithms

### Supported Algorithms

**FedAvg (Federated Averaging)**  
The fundamental algorithm of federated learning:

- Weighted average of model parameters  
- Optimized implementation for deep neural networks  
- Baseline for performance comparisons  

**FedProx (Federated Proximal)**  
Robust extension of FedAvg with proximal term:

- Adds regularization to handle data heterogeneity  
- Configurable `proximal_term` parameter (default: 0.01)  
- Better convergence in non-IID scenarios  

**FedYogi**  
Adaptive optimization with momentum control:

- Adaptive server-side optimization algorithm  
- Configurable parameters: `beta1`, `beta2`, `eta`, `tau`  
- Accelerated convergence and improved stability  

**FedAdam**  
Federated variant of the Adam optimizer:

- Adapts Adam to the federated context  
- Same parameters as FedYogi with different implementation  
- Effective for models with many parameters  

### Algorithm Configuration

Algorithms are configured in the `cfg/config.json` file:

```json
{
  "fl_algorithm": "fedyogi",
  "proximal_term": 0.01,
  "beta1": 0.9,
  "beta2": 0.99,
  "eta": 0.01,
  "tau": 0.001
}
```

---

## Testing Systems

### Main System (start_server.py + start_clients.py)

Architecture based on separate processes for maximum control:

**Features:**

- Granular control of server and clients  
- Modern and flexible system  
- Support for all aggregation policies  
- Advanced configurable resource simulation  

**Usage:**

```bash
# Terminal 1: Server
./venv/Scripts/python.exe start_server.py --policy uniform

# Terminal 2: Client
./venv/Scripts/python.exe start_clients.py --mode resource --policy power
```

#### Main System - Web App version  
A **Web Application based on Gradio** has been developed to easily modify the configuration file and view results of both server and clients, without using the terminal. The interface is **user-friendly** and offers multiple visualization modes.

To launch the application, simply run the `gradio_app.py` file.  
The available sections are:

- **Configurazione**: easily modify `config.json`, enabling experimentation with different models, clients, and training parameters.  
- ** Avvio Server/Client**: start the server and clients, and select the aggregation type.  
- **Log & Output**: shows server and client terminal output, with a refresh button.  
- **Metriche**: displays client metrics per round in a table, with update functionality.  
- **Dashboard**: provides graphical and tabular visualization to monitor metric progress round by round.  
- **Server Training Metrics**: shows aggregated server metrics both in table and chart form, with a refresh button.  

### Automatic Policy Testing System

System for automatic testing of multiple aggregation policies:

**Features:**

- Fully automated with integrated server and clients  
- Batch testing of all available policies  
- Diversified resource simulation per client  
- Systematic comparison of aggregation approaches  

**Usage:**

```bash
# Test specific policy
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy power

# Test all policies
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy all
```

### Framework Comparison System

Automatic comparison system between custom framework and FLOWER:

**Features:**

- Systematic performance comparison  
- Modular structured output  
- Detailed reports with comparative metrics  
- Automatic testing of multiple configurations  

**Usage:**

```bash
# Quick test
echo "1" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py

# Complete test
echo "2" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py
```

### Legacy Client System

Simplified system for automatic clients with external server:

**Features:**

- Automatic clients for already running server  
- Compatibility with previous implementations  
- Simple setup for quick testing  

**Usage:**

```bash
# First start server manually
./venv/Scripts/python.exe start_server.py --policy uniform

# Then automatic clients
./venv/Scripts/python.exe run_multiple_clients.py
```

---

## System Configuration

### Main Configuration File

The `cfg/config.json` file controls all aspects of the system:

**Connectivity:**

```json
{
  "ip_address": "127.0.0.1",
  "port": 5000
}
```

**Training Parameters:**

```json
{
  "global_epoch": 2,
  "local_epoch": 1,
  "num_clients": 4,
  "learning_rate": 1e-5,
  "batch_size": 16
}
```

**Partial Aggregation:**

```json
{
  "partial_aggregation_enabled": true,
  "min_clients_for_aggregation": 2,
  "client_response_timeout": 60
}
```

**FL Algorithm:**

```json
{
  "fl_algorithm": "fedyogi",
  "proximal_term": 0.01,
  "beta1": 0.9,
  "beta2": 0.99,
  "eta": 0.01,
  "tau": 0.001
}
```

### Directory Structure

**Framework Comparison Output:**

```
framework_comparison_outputs/
├── results/     # CSV files with detailed metrics
├── logs/        # Full execution logs
└── summaries/   # Comparative summary reports
```

**System Logs:**

```
logs/
└── MMDD/
    ├── FL-Server-LOG/  # Server logs
    └── FL-Client-LOG/  # Client logs
```

**CSV Results:**

```
csv/
└── MMDD/              # Training metrics by date
```

---

## Monitoring and Debugging

### Logging System

The system implements structured multi-level logging:

- **Server Logs**: Aggregation decisions, client selection, global metrics  
- **Client Logs**: Local training, used resources, communication  
- **Framework Logs**: Performance comparisons, system errors  

### Monitored Metrics

**Training Metrics:**

- Training and validation loss per round  
- Accuracy, Precision, Recall, F1-Score  
- Convergence time  
- Number of participating clients per round  

**System Metrics:**

- Usage of simulated resources  
- Communication and computation times  
- Client success rate  
- Distribution of aggregation weights  
