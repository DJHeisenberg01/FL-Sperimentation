"""
Comprehensive Testing Framework: Custom FL vs FLOWER
Runs systematic comparison between frameworks and exports results to CSV
"""
import os
import sys
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import itertools
import traceback

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'flower_implementation'))

# Try to import framework runners
try:
    from flower_implementation.custom_runner import run_custom_experiment
    CUSTOM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Custom framework not available: {e}")
    CUSTOM_AVAILABLE = False

try:
    from flower_implementation.flower_runner import run_flower_experiment
    FLOWER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FLOWER framework not available: {e}")
    FLOWER_AVAILABLE = False


class FrameworkComparison:
    """Main class for comparing Custom FL vs FLOWER frameworks"""
    
    def __init__(self, dataset_path: str, output_dir: str = "framework_comparison_outputs"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.results = []
        
        # Create organized directory structure
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, "results")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.summaries_dir = os.path.join(output_dir, "summaries")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        
        # Files with organized paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(self.results_dir, f"framework_comparison_{timestamp}.csv")
        self.log_file = os.path.join(self.logs_dir, f"framework_comparison_{timestamp}.log")
        self.summary_file = os.path.join(self.summaries_dir, f"framework_comparison_{timestamp}_summary.txt")
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("=" * 60)
        self.logger.info("🧪 FRAMEWORK COMPARISON TEST SESSION STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Output Directory: {output_dir}")
        self.logger.info(f"📊 Results Directory: {self.results_dir}")
        self.logger.info(f"📋 Logs Directory: {self.logs_dir}")
        self.logger.info(f"📝 Summaries Directory: {self.summaries_dir}")
        self.logger.info(f"Results File: {self.results_file}")
        self.logger.info(f"Log File: {self.log_file}")
        self.logger.info(f"Summary File: {self.summary_file}")
        
        print(f"Framework Comparison initialized")
        print(f"Dataset: {dataset_path}")
        print(f"📁 Organized Output Structure:")
        print(f"   📊 Results: {self.results_dir}")
        print(f"   📋 Logs: {self.logs_dir}")
        print(f"   📝 Summaries: {self.summaries_dir}")
        print(f"Log file: {self.log_file}")
        
    def setup_logging(self):
        """Setup logging system for framework comparison"""
        # Create logger
        self.logger = logging.getLogger("FrameworkComparison")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '🧪 %(levelname)s | %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        print(f"Results will be saved to: {self.results_file}")
        print(f"Custom Framework Available: {CUSTOM_AVAILABLE}")
        print(f"FLOWER Framework Available: {FLOWER_AVAILABLE}")
    
    def define_test_scenarios(self) -> List[Dict[str, Any]]:
        """Define comprehensive test scenarios"""
        scenarios = []
        
        # Base scenarios
        base_configs = [
            {"num_clients": 4, "num_rounds": 2},
            {"num_clients": 4, "num_rounds": 3},
            {"num_clients": 6, "num_rounds": 2},
            {"num_clients": 8, "num_rounds": 2},
        ]
        
        # Custom framework configurations
        custom_configs = [
            {"policy": "uniform", "fl_algorithm": "fedavg"},
            {"policy": "uniform", "fl_algorithm": "fedyogi"},
            {"policy": "power", "fl_algorithm": "fedavg"},
            {"policy": "power", "fl_algorithm": "fedyogi"},
        ]
        
        # FLOWER configurations
        flower_configs = [
            {"strategy": "fedavg"},
            {"strategy": "fedprox"},
        ]
        
        # Generate Custom framework scenarios
        if CUSTOM_AVAILABLE:
            for base, custom in itertools.product(base_configs, custom_configs):
                scenario = {
                    "framework": "Custom",
                    "test_name": f"Custom_{custom['policy']}_{custom['fl_algorithm']}_{base['num_clients']}c_{base['num_rounds']}r",
                    **base,
                    **custom
                }
                scenarios.append(scenario)
        
        # Generate FLOWER scenarios
        if FLOWER_AVAILABLE:
            for base, flower in itertools.product(base_configs, flower_configs):
                scenario = {
                    "framework": "FLOWER",
                    "test_name": f"FLOWER_{flower['strategy']}_{base['num_clients']}c_{base['num_rounds']}r",
                    **base,
                    **flower
                }
                scenarios.append(scenario)
        
        print(f"Generated {len(scenarios)} test scenarios")
        return scenarios
    
    def run_single_experiment(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment scenario"""
        experiment_start = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 STARTING EXPERIMENT: {scenario['test_name']}")
        self.logger.info(f"Framework: {scenario['framework']}")
        self.logger.info(f"Parameters: {scenario}")
        self.logger.info("=" * 60)
        
        print(f"\n{'='*60}")
        print(f"Running: {scenario['test_name']}")
        print(f"Framework: {scenario['framework']}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            if scenario["framework"] == "Custom" and CUSTOM_AVAILABLE:
                self.logger.info("📊 Running Custom Framework experiment...")
                results = run_custom_experiment(
                    dataset_path=self.dataset_path,
                    policy=scenario["policy"],
                    fl_algorithm=scenario["fl_algorithm"],
                    num_clients=scenario["num_clients"],
                    num_rounds=scenario["num_rounds"]
                )
                self.logger.info(f"✅ Custom Framework experiment completed: {results}")
            
            elif scenario["framework"] == "FLOWER" and FLOWER_AVAILABLE:
                self.logger.info("🌸 Running FLOWER Framework experiment...")
                results = run_flower_experiment(
                    dataset_path=self.dataset_path,
                    strategy=scenario["strategy"],
                    num_clients=scenario["num_clients"],
                    num_rounds=scenario["num_rounds"]
                )
                self.logger.info(f"✅ FLOWER Framework experiment completed: {results}")
            
            else:
                error_msg = f"Framework {scenario['framework']} not available"
                self.logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Add scenario info to results
            results.update({
                "test_name": scenario["test_name"],
                "timestamp": datetime.now().isoformat(),
                "experiment_duration": time.time() - start_time
            })
            
            # Add specific parameters based on framework
            if scenario["framework"] == "Custom":
                results.update({
                    "policy": scenario["policy"],
                    "fl_algorithm": scenario["fl_algorithm"]
                })
            elif scenario["framework"] == "FLOWER":
                results.update({
                    "strategy": scenario["strategy"]
                })
            
            experiment_duration = time.time() - experiment_start
            self.logger.info(f"🎉 EXPERIMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"⏱️ Total Duration: {experiment_duration:.2f}s")
            self.logger.info(f"📈 Final Results: Loss={results.get('final_loss', 'N/A')}, Acc={results.get('final_accuracy', 'N/A')}, F1={results.get('final_f1', 'N/A')}")
            
            print(f"✅ Experiment completed successfully")
            print(f"   Final Loss: {results.get('final_loss', 'N/A')}")
            print(f"   Final Accuracy: {results.get('final_accuracy', 'N/A'):.4f}")
            print(f"   Training Time: {results.get('training_time', experiment_duration):.2f}s")
            
            return results
            
        except Exception as e:
            experiment_duration = time.time() - experiment_start
            error_msg = f"Experiment failed: {str(e)}"
            self.logger.error(f"❌ EXPERIMENT FAILED")
            self.logger.error(f"❌ Error: {error_msg}")
            self.logger.error(f"⏱️ Failed after: {experiment_duration:.2f}s")
            self.logger.error(f"📊 Traceback: {traceback.format_exc()}")
            
            print(f"❌ Experiment failed: {str(e)}")
            
            return {
                "test_name": scenario["test_name"],
                "framework": scenario["framework"],
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "experiment_duration": experiment_duration,
                "final_loss": float('inf'),
                "final_accuracy": 0.0,
                "final_f1": 0.0,
                "training_time": experiment_duration
            }
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all defined experiments"""
        scenarios = self.define_test_scenarios()
        
        self.logger.info(f"🚀 Starting comprehensive framework comparison with {len(scenarios)} scenarios")
        
        print(f"\n🚀 Starting comprehensive framework comparison")
        print(f"Total experiments to run: {len(scenarios)}")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📊 Experiment {i}/{len(scenarios)}")
            
            result = self.run_single_experiment(scenario)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Add delay between experiments to avoid conflicts
            if i < len(scenarios):
                self.logger.info("⏳ Waiting 10 seconds before next experiment...")
                print("⏳ Waiting 10 seconds before next experiment...")
                time.sleep(10)
        
        self.logger.info("🎉 ALL EXPERIMENTS COMPLETED!")
        self.logger.info(f"📁 Results saved to: {self.results_file}")
        
        print(f"\n🎉 All experiments completed!")
        print(f"Results saved to: {self.results_file}")
        
        return pd.DataFrame(self.results)
    
    def save_results(self):
        """Save current results to CSV in organized structure"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
            self.logger.info(f"💾 Results saved to {self.results_file} ({len(self.results)} experiments)")
            print(f"� Results saved to {self.results_file}")
    
    def generate_summary_report(self) -> str:
        """Generate summary report of results"""
        if not self.results:
            return "No results to summarize"
        
        df = pd.DataFrame(self.results)
        
        # Filter successful experiments
        successful = df[df['success'] == True]
        
        if len(successful) == 0:
            return "No successful experiments to analyze"
        
        report = []
        report.append("📊 FRAMEWORK COMPARISON SUMMARY")
        report.append("=" * 50)
        
        # Overall stats
        report.append(f"Total Experiments: {len(df)}")
        report.append(f"Successful: {len(successful)}")
        report.append(f"Failed: {len(df) - len(successful)}")
        report.append("")
        
        # Framework comparison
        if 'framework' in successful.columns:
            framework_stats = successful.groupby('framework').agg({
                'final_loss': ['mean', 'std', 'min'],
                'final_accuracy': ['mean', 'std', 'max'],
                'final_f1': ['mean', 'std', 'max'],
                'training_time': ['mean', 'std', 'min']
            }).round(4)
            
            report.append("🏆 FRAMEWORK PERFORMANCE COMPARISON")
            report.append("-" * 40)
            report.append(str(framework_stats))
            report.append("")
        
        # Best performing configurations
        if len(successful) > 0:
            best_accuracy = successful.loc[successful['final_accuracy'].idxmax()]
            best_f1 = successful.loc[successful['final_f1'].idxmax()]
            best_loss = successful.loc[successful['final_loss'].idxmin()]
            
            report.append("🥇 BEST PERFORMING CONFIGURATIONS")
            report.append("-" * 40)
            report.append(f"Best Accuracy: {best_accuracy['test_name']} ({best_accuracy['final_accuracy']:.4f})")
            report.append(f"Best F1 Score: {best_f1['test_name']} ({best_f1['final_f1']:.4f})")
            report.append(f"Best Loss: {best_loss['test_name']} ({best_loss['final_loss']:.4f})")
            report.append("")
        
        # Framework availability
        report.append("🔧 FRAMEWORK AVAILABILITY")
        report.append("-" * 40)
        report.append(f"Custom Framework: {'✅ Available' if CUSTOM_AVAILABLE else '❌ Not Available'}")
        report.append(f"FLOWER Framework: {'✅ Available' if FLOWER_AVAILABLE else '❌ Not Available'}")
        
        return "\n".join(report)
    
    def run_quick_test(self) -> pd.DataFrame:
        """Run a quick test with minimal scenarios"""
        print("🚀 Running Quick Test (minimal scenarios)")
        
        quick_scenarios = []
        
        
        
        if FLOWER_AVAILABLE:
            quick_scenarios.append({
                "framework": "FLOWER",
                "test_name": "Quick_FLOWER_fedavg",
                "num_clients": 4,
                "num_rounds": 2,
                "strategy": "fedavg"
            })
        
        # Quick test scenarios
        if CUSTOM_AVAILABLE:
            quick_scenarios.append({
                "framework": "Custom",
                "test_name": "Quick_Custom_uniform_fedavg",
                "num_clients": 4,
                "num_rounds": 2,
                "policy": "uniform",
                "fl_algorithm": "fedavg"
            })

        for scenario in quick_scenarios:
            result = self.run_single_experiment(scenario)
            self.results.append(result)
        
        self.save_results()
        return pd.DataFrame(self.results)


def main():
    """Main execution function"""
    # Configuration
    dataset_path = "dataset/Cropped_ROI"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please ensure the dataset is available at the specified path")
        return
    
    # Initialize comparison
    comparison = FrameworkComparison(dataset_path)
    
    # Choose test type
    test_type = input("\nChoose test type:\n1. Quick Test (few scenarios)\n2. Full Test (all scenarios)\nEnter choice (1 or 2): ").strip()
    
    if test_type == "1":
        print("Running Quick Test...")
        results_df = comparison.run_quick_test()
    else:
        print("Running Full Test...")
        results_df = comparison.run_all_experiments()
    
    # Generate and display summary
    summary = comparison.generate_summary_report()
    print(f"\n{summary}")
    
    # Save summary to organized file
    with open(comparison.summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n📁 Files saved in organized structure:")
    print(f"📊 Summary: {comparison.summary_file}")
    print(f"� Detailed results: {comparison.results_file}")
    print(f"📝 Logs: {comparison.log_file}")


if __name__ == "__main__":
    main()
