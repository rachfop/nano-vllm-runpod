#!/usr/bin/env python3
"""
Simplified test script for nano-vllm Runpod deployment configuration.
This script validates the basic configuration without importing problematic modules.
"""

import os
import sys
import json
from pathlib import Path

def test_configuration_files():
    """Test that all configuration files exist and are valid"""
    print("Testing configuration files...")
    
    base_path = Path(__file__).parent
    
    # Test files exist
    files_to_check = [
        "Dockerfile",
        ".runpod/hub.json",
        "builder/requirements.txt",
        "handler.py",
        "pyproject.toml"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    # Test JSON validity
    try:
        with open(base_path / ".runpod/hub.json", "r") as f:
            hub_config = json.load(f)
        print("✓ hub.json is valid JSON")
        
        # Check required fields
        required_fields = ["title", "description", "type", "config"]
        for field in required_fields:
            if field in hub_config:
                print(f"✓ hub.json has required field: {field}")
            else:
                print(f"✗ hub.json missing required field: {field}")
                all_exist = False
                
    except json.JSONDecodeError as e:
        print(f"✗ hub.json is invalid JSON: {e}")
        all_exist = False
    
    return all_exist

def test_dockerfile():
    """Basic Dockerfile validation"""
    print("Testing Dockerfile...")
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    if not dockerfile_path.exists():
        print("✗ Dockerfile not found")
        return False
    
    with open(dockerfile_path, "r") as f:
        content = f.read()
    
    # Check for required components
    checks = [
        ("CUDA base image", "nvidia/cuda:12.1.0-base-ubuntu22.04" in content),
        ("Python installation", "python3-pip" in content),
        ("runpod dependency", "runpod" in content),
        ("Handler execution", "handler.py" in content),
        ("Working directory", "WORKDIR /app" in content)
    ]
    
    all_good = True
    for check_name, condition in checks:
        if condition:
            print(f"✓ Dockerfile has {check_name}")
        else:
            print(f"✗ Dockerfile missing {check_name}")
            all_good = False
    
    return all_good

def test_handler_syntax():
    """Test handler Python syntax without importing"""
    print("Testing handler syntax...")
    
    handler_path = Path(__file__).parent / "handler.py"
    if not handler_path.exists():
        print("✗ handler.py not found")
        return False
    
    try:
        with open(handler_path, "r") as f:
            code = f.read()
        
        # Basic syntax check
        compile(code, str(handler_path), 'exec')
        print("✓ handler.py has valid Python syntax")
        
        # Check for required components
        required_components = [
            "import runpod",
            "class NanoVLLMEngine",
            "async def handler",
            "runpod.serverless.start"
        ]
        
        all_present = True
        for component in required_components:
            if component in code:
                print(f"✓ handler.py contains: {component}")
            else:
                print(f"✗ handler.py missing: {component}")
                all_present = False
        
        return all_present
        
    except SyntaxError as e:
        print(f"✗ handler.py has syntax error: {e}")
        return False

def test_pyproject_toml():
    """Test pyproject.toml configuration"""
    print("Testing pyproject.toml...")
    
    toml_path = Path(__file__).parent / "pyproject.toml"
    if not toml_path.exists():
        print("✗ pyproject.toml not found")
        return False
    
    try:
        # Simple TOML parsing (basic validation)
        with open(toml_path, "r") as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            "[project]",
            "dependencies",
            "runpod",
            "torch>=2.4.0"
        ]
        
        all_present = True
        for section in required_sections:
            if section in content:
                print(f"✓ pyproject.toml contains: {section}")
            else:
                print(f"✗ pyproject.toml missing: {section}")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"✗ pyproject.toml validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== nano-vLLM Runpod Deployment Configuration Test ===\n")
    print("Note: This test validates configuration without importing nano-vllm modules")
    print("due to Python 3.12+ compatibility issues.\n")
    
    tests = [
        ("Configuration Files", test_configuration_files),
        ("Dockerfile Validation", test_dockerfile),
        ("Handler Syntax", test_handler_syntax),
        ("pyproject.toml", test_pyproject_toml)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All configuration tests passed!")
        print("\nNext steps:")
        print("1. Test with Python 3.10 or 3.11 (nano-vllm compatibility)")
        print("2. Install dependencies: pip install -e .")
        print("3. Test with a local model download")
        print("4. Build Docker image: docker build -t nano-vllm-runpod .")
        print("5. Deploy to Runpod Hub")
        print("\nNote: For production deployment, use Python 3.10 or 3.11")
        print("as nano-vllm has compatibility issues with Python 3.12+")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
