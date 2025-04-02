#!/usr/bin/env python3
"""
Environment Setup Utility for RAG-MCP Pipeline Research

This script helps verify and set up the necessary environment for the research project.
It checks Python version, required packages, and provides guidance for next steps.

This project uses free open-source models from Hugging Face instead of paid API services.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version meets the minimum requirement."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"ERROR: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✓ Python version {current_version[0]}.{current_version[1]} meets requirements.")
    return True


def check_pip():
    """Verify pip is installed and accessible."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ pip is installed and accessible.")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: pip is not installed or not accessible.")
        return False


def setup_virtual_environment():
    """Set up a virtual environment if not already created."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists.")
        return True
    
    try:
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Activation instructions
        if platform.system() == "Windows":
            print("\nTo activate the virtual environment, run:")
            print("    venv\\Scripts\\activate")
        else:
            print("\nTo activate the virtual environment, run:")
            print("    source venv/bin/activate")
            
        print("✓ Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to create virtual environment: {e}")
        return False


def install_base_packages():
    """Install base packages needed for the project."""
    # Updated packages to use Hugging Face models instead of OpenAI
    packages = [
        "transformers",        # Hugging Face Transformers for LLMs
        "torch",               # PyTorch as the backend for models
        "sentence-transformers", # For embeddings and semantic search
        "pytest",              # For testing
        "requests",            # HTTP requests
        "jupyter",             # Jupyter notebooks
        "fastapi",             # For MCP server
        "uvicorn",             # ASGI server for FastAPI
        "tqdm"                 # Progress bars
    ]
    
    print("\nWould you like to install the recommended packages?")
    print("The following packages will be installed:")
    for pkg in packages:
        print(f"  - {pkg}")
    
    print("\nNOTE: These are free and open-source alternatives to paid API services.")
    print("Models will download automatically when needed (1-2GB of disk space required).")
    
    response = input("\nInstall packages? (y/n): ").strip().lower()
    
    if response == "y":
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        try:
            print("\nInstalling packages (this may take a few minutes)...")
            subprocess.run(cmd, check=True)
            print("✓ Base packages installed successfully.")
            
            # Create requirements.txt
            with open("requirements.txt", "w") as f:
                for pkg in packages:
                    f.write(f"{pkg}\n")
            print("✓ requirements.txt created.")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install packages: {e}")
            return False
    else:
        print("Skipped package installation.")
        return True


def verify_gpu():
    """Check if GPU is available for faster model inference."""
    try:
        # First check if torch is installed
        subprocess.run([sys.executable, "-c", "import torch"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Then check if CUDA is available
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        has_gpu = result.stdout.strip() == "True"
        
        if has_gpu:
            print("✓ GPU detected! Models will run much faster.")
        else:
            print("ℹ No GPU detected. Models will run on CPU (slower but still functional).")
            
        return True
    except:
        print("ℹ Could not verify GPU. Models will run on CPU.")
        return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n=== NEXT STEPS ===")
    print("1. Activate the virtual environment")
    print("   • Windows: venv\\Scripts\\activate")
    print("   • macOS/Linux: source venv/bin/activate")
    print("2. Start with Module 0:")
    print("   • Read docs/modules/module_0/README.md")
    print("   • Run the examples in docs/modules/module_0/")
    print("3. Complete the practical exercises in Module 0")
    print("\nℹ The first time you run scripts with models, they will download automatically.")
    print("  This will take some time and require 1-2GB of disk space.")
    print("\nHappy learning! 🚀")


def main():
    """Main function to run all setup checks and procedures."""
    print("=== RAG-MCP Pipeline Research Environment Setup ===\n")
    print("This setup uses FREE open-source models instead of paid API services\n")
    
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    if not setup_virtual_environment():
        return False
    
    if not install_base_packages():
        return False
    
    verify_gpu()
    
    print_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 