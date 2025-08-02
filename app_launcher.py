#!/usr/bin/env python3
"""
RAGFlow Insurance Policy System - App Launcher

Simple launcher script for the Streamlit web interface with setup validation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  RAGFlow Web Interface Launcher                 ║
    ║                     Insurance Policy Assistant                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    🏥 Starting RAGFlow Insurance Policy Assistant
    🌐 Web Interface powered by Streamlit
    🦙 Local Llama 3 Integration
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages, check=True)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_system_setup():
    """Check if the RAGFlow system is properly configured"""
    print("\n🔧 Checking system configuration...")
    
    # Check if config file exists
    if not Path("config.py").exists():
        print("❌ config.py not found")
        return False
    
    try:
        from config import RAGFlowConfig
        config = RAGFlowConfig()
        
        # Basic validation
        if config.LLM_PROVIDER == "openai" and not config.get_openai_api_key():
            print("⚠️  OpenAI API key not configured")
        elif config.LLM_PROVIDER == "anthropic" and not config.get_anthropic_api_key():
            print("⚠️  Anthropic API key not configured")
        elif config.LLM_PROVIDER in ["local", "ollama"]:
            print(f"✅ Local model configured: {config.LOCAL_MODEL_TYPE}")
        
        # Check PDF file
        if not Path(config.PDF_FILE).exists():
            print(f"⚠️  PDF file not found: {config.PDF_FILE}")
        else:
            print(f"✅ PDF file found: {config.PDF_FILE}")
        
        print("✅ System configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Starting Streamlit web interface...")
    print("📍 The application will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*60)
    print("📝 Usage Tips:")
    print("   • Ask questions about your insurance policy")
    print("   • Use the example queries in the sidebar")
    print("   • Check the system metrics for performance")
    print("   • Enable debug mode for detailed information")
    print("="*60)
    print("\n⏹️  Press Ctrl+C to stop the application\n")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_interface.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 RAGFlow application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start Streamlit: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        return False
    
    return True

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages and try again.")
        return 1
    
    # Check system setup
    if not check_system_setup():
        print("\n⚠️  System setup issues detected.")
        print("💡 Quick fixes:")
        print("   • Run: python setup_llama3.py")
        print("   • Check your .env configuration")
        print("   • Ensure all files are in place")
        
        response = input("\n❓ Continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("👋 Setup cancelled. Please fix the issues and try again.")
            return 1
    
    # Launch application
    print("\n✨ All checks passed! Launching RAGFlow...")
    time.sleep(1)
    
    if launch_streamlit():
        print("\n✅ RAGFlow session completed successfully!")
        return 0
    else:
        print("\n❌ Failed to launch RAGFlow")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 