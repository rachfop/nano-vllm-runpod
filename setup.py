#!/usr/bin/env python3
"""
Setup script for nano-vLLM Runpod Edition fork.
This script helps initialize and configure the fork for deployment.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10 and version.minor < 12:
        print(f"✓ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} is not recommended")
        print("Please use Python 3.10 or 3.11 for nano-vLLM compatibility")
        return False

def install_dependencies():
    """Install package dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                        check=True, capture_output=True, text=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def validate_configuration():
    """Run configuration validation"""
    print("Validating configuration...")
    try:
        result = subprocess.run([sys.executable, "test_config.py"], 
                                check=True, capture_output=True, text=True)
        print("✓ Configuration validation passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def create_gitignore():
    """Create .gitignore file if it doesn't exist"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Logs
*.log
logs/

# Model files
*.bin
*.safetensors
*.pth
models/
cache/

# Runpod
.runpod/secrets
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore")
    else:
        print("✓ .gitignore already exists")

def update_hub_config():
    """Update hub.json with repository information"""
    print("Updating hub configuration...")
    
    hub_path = Path(".runpod/hub.json")
    if hub_path.exists():
        try:
            with open(hub_path, "r") as f:
                config = json.load(f)
            
            # Update with your repository info
            config["repository"] = "https://github.com/your-username/nano-vllm-runpod"
            config["documentation"] = "https://github.com/your-username/nano-vllm-runpod/blob/main/README.md"
            
            with open(hub_path, "w") as f:
                json.dump(config, f, indent=2)
            
            print("✓ Updated hub.json with repository information")
            return True
        except Exception as e:
            print(f"✗ Failed to update hub.json: {e}")
            return False
    else:
        print("✗ hub.json not found")
        return False

def create_docker_compose():
    """Create docker-compose.yml for local testing"""
    compose_content = """version: '3.8'

services:
  nano-vllm-runpod:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=Qwen/Qwen3-8B
      - TENSOR_PARALLEL_SIZE=1
      - MAX_MODEL_LEN=4096
      - GPU_MEMORY_UTILIZATION=0.9
      - MAX_CONCURRENCY=30
    volumes:
      - ./cache:/runpod-volume/huggingface-cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    """
    
    compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        with open(compose_path, "w") as f:
            f.write(compose_content)
        print("✓ Created docker-compose.yml for local testing")
    else:
        print("✓ docker-compose.yml already exists")

def main():
    """Main setup function"""
    print("=== nano-vLLM Runpod Edition Setup ===")
    print("Setting up your fork for deployment...\n")
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Validate Configuration", validate_configuration),
        ("Create .gitignore", create_gitignore),
        ("Update Hub Config", update_hub_config),
        ("Create Docker Compose", create_docker_compose)
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"✗ {step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n=== Setup Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for step_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {step_name}")
    
    print(f"\nOverall: {passed}/{total} steps completed successfully")
    
    if passed == total:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Configure your repository URL in .runpod/hub.json")
        print("2. Build Docker image: docker build -t nano-vllm-runpod .")
        print("3. Test locally: docker-compose up")
        print("4. Deploy to Runpod Hub")
        print("\nFor deployment help, see DEPLOYMENT.md")
    else:
        print(f"\n⚠️  {total - passed} steps failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())