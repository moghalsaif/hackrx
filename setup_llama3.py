#!/usr/bin/env python3
"""
Llama 3 Setup Script for RAGFlow Insurance Policy System

This script helps you set up Llama 3 for local inference in the RAGFlow system.
It provides multiple setup options including Hugging Face Transformers and Ollama.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional
import json
import urllib.request
import shutil

def print_banner():
    """Print setup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                     RAGFlow Llama 3 Setup                        ║
    ║               Local LLM Configuration Assistant                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    🦙 Setting up Llama 3 for local inference
    📋 Multiple deployment options available
    ⚡ Optimized for insurance policy analysis
    """
    print(banner)

def check_system_requirements():
    """Check system requirements for running Llama 3"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️  Warning: Llama 3 8B requires at least 8GB RAM for optimal performance")
        
    except ImportError:
        print("⚠️  Cannot check memory - psutil not installed")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"🚀 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("💻 No CUDA GPU detected - CPU inference will be used")
            
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
    
    return True

def setup_ollama():
    """Setup Llama 3 using Ollama"""
    print("\n🦙 Setting up Llama 3 with Ollama...")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Ollama found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama not found. Please install Ollama first:")
        print("   Visit: https://ollama.ai/download")
        print("   Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    
    # Pull Llama 3 model
    print("📥 Downloading Llama 3 model (this may take a while)...")
    
    try:
        # Try to pull llama3 (8B model)
        subprocess.run(["ollama", "pull", "llama3"], check=True)
        print("✅ Llama 3 8B model downloaded successfully")
        
        # Test the model
        print("🧪 Testing model...")
        result = subprocess.run(
            ["ollama", "run", "llama3", "Hello! Please respond with 'Test successful.'"],
            capture_output=True, text=True, timeout=30
        )
        
        if "test successful" in result.stdout.lower():
            print("✅ Model test successful")
        else:
            print("⚠️  Model test completed but response unclear")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download model: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Model test timed out, but download may have succeeded")
        return True

def setup_transformers(model_path: Optional[str] = None):
    """Setup Llama 3 using Hugging Face Transformers"""
    print("\n🤗 Setting up Llama 3 with Hugging Face Transformers...")
    
    if model_path is None:
        model_path = Path("models/llama3")
    else:
        model_path = Path(model_path)
    
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Model will be saved to: {model_path.absolute()}")
    
    # Check if transformers is installed
    try:
        import transformers
        print(f"✅ Transformers found: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers>=4.36.0"], check=True)
    
    # Instructions for manual download
    print("\n📋 Manual Setup Instructions:")
    print("1. Visit: https://huggingface.co/meta-llama/Llama-3-8B-Instruct")
    print("2. Request access to the model (requires Meta approval)")
    print("3. Once approved, download the model files to:", model_path.absolute())
    print("4. Or use the Hugging Face CLI:")
    print(f"   huggingface-cli download meta-llama/Llama-3-8B-Instruct --local-dir {model_path}")
    
    # Alternative: Use the download link provided by user
    print("\n🔗 Alternative: Using provided download link...")
    print("The download link you provided appears to be from Meta's official distribution.")
    print("You can use it to download the model files manually.")
    
    # Check if model files exist
    if any(model_path.glob("*.bin")) or any(model_path.glob("*.safetensors")):
        print("✅ Model files detected in the specified directory")
        return True
    else:
        print("⚠️  No model files found. Please download the model manually.")
        return False

def setup_llamacpp():
    """Setup Llama 3 using llama.cpp"""
    print("\n🔧 Setting up Llama 3 with llama.cpp...")
    
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python found")
    except ImportError:
        print("❌ llama-cpp-python not installed. Installing...")
        # Install with CUDA support if available
        try:
            import torch
            if torch.cuda.is_available():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python[cuda]"
                ], check=True)
            else:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python"
                ], check=True)
        except ImportError:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python"
            ], check=True)
    
    print("📋 For llama.cpp setup:")
    print("1. Download a GGUF format Llama 3 model")
    print("2. Visit: https://huggingface.co/models?search=llama-3+gguf")
    print("3. Place the .gguf file in: models/llama3/")
    print("4. Update LOCAL_MODEL_PATH in your .env file")
    
    return True

def create_env_config(setup_type: str, model_path: Optional[str] = None):
    """Create environment configuration for the selected setup"""
    print(f"\n⚙️  Creating configuration for {setup_type} setup...")
    
    env_config = {
        "ollama": {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "llama3",
            "LOCAL_MODEL_TYPE": "ollama", 
            "OLLAMA_MODEL_NAME": "llama3",
            "OLLAMA_BASE_URL": "http://localhost:11434"
        },
        "transformers": {
            "LLM_PROVIDER": "local",
            "LLM_MODEL": "llama3",
            "LOCAL_MODEL_TYPE": "transformers",
            "LOCAL_MODEL_PATH": model_path or "models/llama3",
            "LOCAL_MODEL_DEVICE": "auto",
            "LOCAL_MODEL_PRECISION": "fp16"
        },
        "llamacpp": {
            "LLM_PROVIDER": "local",
            "LLM_MODEL": "llama3",
            "LOCAL_MODEL_TYPE": "llamacpp",
            "LOCAL_MODEL_PATH": model_path or "models/llama3/model.gguf",
            "LOCAL_MODEL_DEVICE": "auto"
        }
    }
    
    config = env_config.get(setup_type, {})
    
    # Read existing .env or create new one
    env_file = Path(".env")
    existing_config = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    existing_config[key] = value
    
    # Update with new config
    existing_config.update(config)
    
    # Write updated config
    with open(env_file, 'w') as f:
        f.write("# RAGFlow Configuration - Auto-generated by setup script\n\n")
        f.write("# LLM Configuration\n")
        for key, value in existing_config.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ Configuration saved to .env")
    print(f"🔧 LLM Provider: {config.get('LLM_PROVIDER', 'unknown')}")
    print(f"🤖 Model Type: {config.get('LOCAL_MODEL_TYPE', 'unknown')}")

def test_setup():
    """Test the Llama 3 setup"""
    print("\n🧪 Testing Llama 3 setup...")
    
    try:
        # Import and test the RAGFlow system
        from config import RAGFlowConfig
        from llm_reasoner import LLMReasoner
        
        config = RAGFlowConfig()
        reasoner = LLMReasoner(config)
        
        print("✅ RAGFlow system loaded successfully")
        print("🎉 Llama 3 setup complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        print("Please check the configuration and try again.")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Set up Llama 3 for RAGFlow")
    parser.add_argument("--method", choices=["ollama", "transformers", "llamacpp"], 
                       help="Setup method")
    parser.add_argument("--model-path", help="Path for model files (transformers/llamacpp only)")
    parser.add_argument("--test-only", action="store_true", help="Only test existing setup")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.test_only:
        test_setup()
        return
    
    if not check_system_requirements():
        return
    
    # Interactive setup if no method specified
    if not args.method:
        print("\n🔧 Available setup methods:")
        print("1. Ollama (Recommended - Easy setup)")
        print("2. Hugging Face Transformers (More control)")
        print("3. llama.cpp (Efficient inference)")
        
        choice = input("\nSelect setup method (1-3): ").strip()
        
        method_map = {"1": "ollama", "2": "transformers", "3": "llamacpp"}
        args.method = method_map.get(choice)
        
        if not args.method:
            print("❌ Invalid choice")
            return
    
    # Run setup based on method
    success = False
    
    if args.method == "ollama":
        success = setup_ollama()
    elif args.method == "transformers":
        success = setup_transformers(args.model_path)
    elif args.method == "llamacpp":
        success = setup_llamacpp()
    
    if success:
        create_env_config(args.method, args.model_path)
        test_setup()
    else:
        print("❌ Setup failed. Please check the errors above and try again.")

if __name__ == "__main__":
    main() 