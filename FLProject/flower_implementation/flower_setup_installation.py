"""
Setup script for FLOWER framework comparison
Updated version with correct paths for the new project structure
"""
import subprocess
import sys
import os

def install_flower():
    """Install FLOWER framework and dependencies"""
    print("🌸 Installing FLOWER framework...")
    
    try:
        # Updated path - look in current directory (flower_implementation)
        req_file = os.path.join(os.path.dirname(__file__), "requirements_flower.txt")
        
        if os.path.exists(req_file):
            print(f"📋 Installing from requirements file: {req_file}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        else:
            print("📋 Requirements file not found, installing FLOWER directly...")
            # Fallback to direct installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flwr[simulation]==1.11.0"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ray"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        
        print("✅ FLOWER framework installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error installing FLOWER: {e}")
        return False

def verify_installation():
    """Verify both frameworks are working"""
    print("\n🔍 Verifying installations...")
    
    # Test FLOWER
    try:
        import flwr as fl
        print(f"✅ FLOWER framework: OK (version {fl.__version__})")
        flower_ok = True
    except ImportError as e:
        print(f"❌ FLOWER framework: NOT AVAILABLE ({e})")
        flower_ok = False
    
    # Test Ray (required for FLOWER simulation)
    try:
        import ray
        print("✅ Ray framework: OK")
        ray_ok = True
    except ImportError as e:
        print(f"❌ Ray framework: NOT AVAILABLE ({e})")
        ray_ok = False
    
    # Test Custom framework (go up one level to FLProject)
    try:
        # Add parent directory (FLProject) to path
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from federated_server import FederatedServer
        from federated_client import FederatedClient
        print("✅ Custom framework: OK")
        custom_ok = True
    except ImportError as e:
        print(f"❌ Custom framework: NOT AVAILABLE ({e})")
        custom_ok = False
    
    # Test FLOWER implementation components
    try:
        from flower_client import create_flower_client
        from flower_server import FlowerServerConfig
        from flower_dataset import load_flower_data
        print("✅ FLOWER implementation components: OK")
        flower_impl_ok = True
    except ImportError as e:
        print(f"❌ FLOWER implementation components: NOT AVAILABLE ({e})")
        flower_impl_ok = False
    
    return flower_ok and ray_ok, custom_ok, flower_impl_ok

def test_framework_comparison():
    """Test if the framework comparison system works"""
    print("\n🧪 Testing framework comparison system...")
    
    try:
        # Go up one level to FLProject to import the comparison system
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from testing_frameworkcustom_vs_FLOWER import FrameworkComparison
        
        # Test dataset path
        dataset_path = os.path.join(parent_dir, "dataset", "Cropped_ROI")
        if os.path.exists(dataset_path):
            print(f"✅ Dataset found: {dataset_path}")
            
            # Try to initialize (but don't run experiments)
            print("✅ Framework comparison system: Ready")
            return True
        else:
            print(f"❌ Dataset not found: {dataset_path}")
            return False
            
    except ImportError as e:
        print(f"❌ Framework comparison system: NOT AVAILABLE ({e})")
        return False
    except Exception as e:
        print(f"❌ Framework comparison system: ERROR ({e})")
        return False

def check_dataset():
    """Check if required dataset is available"""
    print("\n📊 Checking dataset availability...")
    
    # Go up one level to FLProject
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(parent_dir, "dataset", "Cropped_ROI")
    
    if os.path.exists(dataset_path):
        # Count files in dataset
        try:
            files = []
            for root, dirs, filenames in os.walk(dataset_path):
                files.extend([f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            print(f"✅ Dataset available: {len(files)} images found")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        print("   Please ensure the Cropped_ROI dataset is available")
        return False

def main():
    """Main setup function"""
    print("🚀 FLOWER Framework Setup & Verification")
    print("=" * 50)
    print("Updated for flower_implementation structure")
    print("=" * 50)
    
    # Install FLOWER
    flower_installed = install_flower()
    
    # Verify installations
    flower_ok, custom_ok, flower_impl_ok = verify_installation()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Test framework comparison system
    comparison_ok = test_framework_comparison()
    
    print("\n📊 Setup Summary:")
    print("=" * 30)
    print(f"FLOWER Framework:     {'✅ Ready' if flower_ok else '❌ Not Available'}")
    print(f"Custom Framework:     {'✅ Ready' if custom_ok else '❌ Not Available'}")
    print(f"FLOWER Implementation: {'✅ Ready' if flower_impl_ok else '❌ Not Available'}")
    print(f"Dataset:              {'✅ Ready' if dataset_ok else '❌ Not Available'}")
    print(f"Comparison System:    {'✅ Ready' if comparison_ok else '❌ Not Available'}")
    
    print("\n🎯 Next Steps:")
    if flower_ok and custom_ok and flower_impl_ok and dataset_ok and comparison_ok:
        print("🎉 All systems are ready for framework comparison!")
        print("\nTo run framework comparison:")
        print("   cd ..")  # Go back to FLProject
        print("   python testing_frameworkcustom_vs_FLOWER.py")
        print("\nTo run individual framework tests:")
        print("   # Terminal 1: ./venv/Scripts/python.exe start_server.py --policy uniform")
        print("   # Terminal 2: ./venv/Scripts/python.exe start_clients.py --mode standard")
    else:
        print("⚠️  Some components are missing:")
        if not flower_ok:
            print("   - Install FLOWER: pip install flwr[simulation]==1.11.0")
        if not custom_ok:
            print("   - Check custom framework files in parent directory")
        if not flower_impl_ok:
            print("   - Check FLOWER implementation files in current directory")
        if not dataset_ok:
            print("   - Ensure dataset is available in ../dataset/Cropped_ROI/")
        if not comparison_ok:
            print("   - Check framework comparison system")

if __name__ == "__main__":
    main()
