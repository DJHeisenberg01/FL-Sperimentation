# 🌸 FLOWER Framework Comparison

## 📋 Overview

This directory contains the implementation of FLOWER framework to compare against your custom FL system.

## 🏗️ Architecture

```
flowe/
├── requirements_flower.txt       # FLOWER dependencies
├── flower_dataset.py            # Dataset handler for FLOWER
├── flower_model.py             # Model implementation (ResNet18)
├── flower_client.py            # FLOWER client implementation
├── flower_server.py            # FLOWER server strategies
├── flower_runner.py            # FLOWER experiment runner
└── custom_runner.py            # Your custom framework runner
```

## 🚀 Quick Start

### 1. Install FLOWER Framework

```bash
# Run setup script
python flower_setup_installation.py

# Or install manually
cd flowe
pip install -r requirements_flower.txt
```

### 2. Run Comparison

```bash
# Full comparison
python testing_frameworkcustom_vs_FLOWER.py

# Quick test
python testing_frameworkcustom_vs_FLOWER.py
# Choose option 1 for quick test
```

## 🧪 Test Scenarios

### Custom Framework Configurations

- **Policies**: `uniform`, `power`
- **FL Algorithms**: `fedavg`, `fedyogi`
- **Combinations**: 4 total configurations

### FLOWER Framework Configurations

- **Strategies**: `FedAvg`, `FedProx`
- **Standard implementations**: 2 total configurations

### Test Parameters

- **Clients**: 4, 6, 8
- **Rounds**: 2, 3
- **Total Scenarios**: ~24 experiments

## 📊 Results Format

Results are saved to `flowe/framework_comparison_YYYYMMDD_HHMMSS.csv`:

```csv
test_name,framework,strategy,num_clients,num_rounds,final_loss,final_accuracy,final_f1,training_time,success
Custom_uniform_fedavg_4c_2r,Custom,uniform_fedavg,4,2,0.58,0.625,0.164,180.5,True
FLOWER_fedavg_4c_2r,FLOWER,fedavg,4,2,0.62,0.598,0.152,165.2,True
```

## 🔧 Implementation Details

### Dataset Compatibility

- Uses same ROI dataset as custom framework
- Identical train/validation splits (StratifiedKFold)
- Same preprocessing and transforms

### Model Consistency

- ResNet18 backbone (identical to custom)
- Same initialization and hyperparameters
- Binary classification (damaged/healthy)

### Fair Comparison

- Same batch size (16)
- Same learning rate (1e-5)
- Same number of local epochs (1)
- Same evaluation metrics

## 📈 Metrics Compared

| Metric            | Description                   | Custom Framework | FLOWER |
| ----------------- | ----------------------------- | ---------------- | ------ |
| **Final Loss**    | CrossEntropy loss on test set | ✅               | ✅     |
| **Accuracy**      | Classification accuracy       | ✅               | ✅     |
| **F1 Score**      | Weighted F1 score             | ✅               | ✅     |
| **Precision**     | Weighted precision            | ✅               | ✅     |
| **Recall**        | Weighted recall               | ✅               | ✅     |
| **Training Time** | Total experiment duration     | ✅               | ✅     |

## 🎯 Key Comparisons

### 1. **Aggregation Strategies**

```
Custom Framework:
- Progressive partial aggregation (2→3→4 clients)
- Resource-aware client selection
- Universal model broadcasting

FLOWER Framework:
- Traditional aggregation (wait for all)
- Simple client sampling
- Standard model distribution
```

### 2. **Client Selection**

```
Custom Framework:
- Power-aware policy (compute power based)
- Reliability-aware policy (stability based)
- Hybrid policies (multi-criteria)

FLOWER Framework:
- Random sampling
- Fraction-based selection
- No resource awareness
```

### 3. **Fault Tolerance**

```
Custom Framework:
- Partial aggregation with minimum clients
- Timeout-based aggregation
- Non-participant model updates

FLOWER Framework:
- Fixed client requirements
- Standard timeout handling
- Participant-only updates
```

## 📊 Analysis Features

### Performance Comparison

- Convergence speed analysis
- Final accuracy comparison
- Training efficiency metrics

### Scalability Testing

- Variable client counts (4, 6, 8)
- Round duration analysis
- Resource utilization

### Robustness Evaluation

- Client dropout simulation
- Network condition variations
- Failure recovery analysis

## 🔍 Debugging

### Common Issues

1. **FLOWER Import Errors**

   ```bash
   pip install flwr[simulation]==1.11.0
   ```

2. **Custom Framework Path Issues**

   ```python
   # Ensure FLProject is in path
   sys.path.append('FLProject')
   ```

3. **Dataset Path Problems**
   ```bash
   # Verify dataset exists
   ls FLProject/dataset/Cropped_ROI/roi_annotation.csv
   ```

### Log Files

- Custom Framework: `FLProject/logs/DDMM/FL-Server-LOG/`
- FLOWER: Console output and CSV results
- Comparison: `flowe/framework_comparison_*.csv`

## 🏆 Expected Results

Based on your custom framework's innovations:

### Advantages of Custom Framework

- **Faster Convergence**: Partial aggregation reduces round time
- **Better Fault Tolerance**: Handles client dropouts gracefully
- **Resource Efficiency**: Smart client selection policies
- **Universal Updates**: All clients receive model updates

### Advantages of FLOWER Framework

- **Standardized Implementation**: Well-tested and documented
- **Community Support**: Large ecosystem and extensions
- **Simulation Features**: Built-in client simulation
- **Research Baseline**: Established benchmark for comparison

## 📝 Citation

If you use this comparison in research:

```bibtex
@misc{custom_flower_comparison,
  title={Progressive Partial Aggregation vs Traditional FL: A Comparative Study},
  author={Your Name},
  year={2025},
  note={Custom FL framework comparison with FLOWER}
}
```

## 🤝 Contributing

To extend the comparison:

1. Add new FLOWER strategies in `flower_server.py`
2. Implement new custom policies in your main framework
3. Update test scenarios in `testing_frameworkcustom_vs_FLOWER.py`
4. Add new metrics in both runners

## 📞 Support

For issues:

1. Check dataset paths and permissions
2. Verify both frameworks are installed
3. Review log files for detailed errors
4. Ensure sufficient system resources (CPU/Memory)
