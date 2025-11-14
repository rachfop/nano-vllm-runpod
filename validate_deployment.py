#!/usr/bin/env python3
"""
Deployment robustness testing and validation script for nano-vLLM.

This script tests the deployment pipeline and validates that all improvements
work correctly across different platforms and configurations.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.model_validator import check_model_compatibility, validate_deployment_config
from nanovllm.utils.platform_compat import get_platform_compat, setup_platform_specific_logging


class DeploymentValidator:
    """Comprehensive deployment validation system."""
    
    def __init__(self):
        self.platform_compat = get_platform_compat()
        self.validation_results = {}
        
    def test_platform_compatibility(self) -> Dict[str, Any]:
        """Test platform compatibility and configuration."""
        logger.info("Testing platform compatibility...")
        
        results = {
            "platform_info": {
                "system": self.platform_compat.system,
                "machine": self.platform_compat.machine,
                "cuda_available": self.platform_compat.is_cuda_available,
                "cuda_version": self.platform_compat.cuda_version,
            },
            "device_check": self._test_device_availability(),
            "memory_check": self._test_memory_limits(),
            "dependency_check": self._test_dependencies(),
        }
        
        self.validation_results["platform"] = results
        return results
    
    def _test_device_availability(self) -> Dict[str, Any]:
        """Test device availability and configuration."""
        try:
            import torch
            device = self.platform_compat.get_optimal_device()
            dtype = self.platform_compat.get_optimal_dtype()
            
            # Test tensor creation on device
            test_tensor = torch.randn(100, 100, device=device, dtype=dtype)
            
            return {
                "status": "pass",
                "device": str(device),
                "dtype": str(dtype),
                "tensor_test": "success"
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    def _test_memory_limits(self) -> Dict[str, Any]:
        """Test memory limits and allocation."""
        try:
            memory_limits = self.platform_compat.get_memory_limits()
            
            # Test allocation within limits
            import torch
            test_size = min(1024**3, memory_limits["max_model_memory"] // 10)  # 1GB or 10% of limit
            
            if self.platform_compat.is_cuda_available:
                # Test GPU memory allocation
                test_tensor = torch.randn(test_size // 4, device="cuda")  # 4 bytes per float32
                del test_tensor
                torch.cuda.empty_cache()
            else:
                # Test CPU memory allocation
                test_tensor = torch.randn(test_size // 4)  # 4 bytes per float32
                del test_tensor
            
            return {
                "status": "pass",
                "memory_limits": memory_limits,
                "allocation_test": "success"
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    def _test_dependencies(self) -> Dict[str, Any]:
        """Test optional dependencies availability."""
        results = {}
        
        # Test flash attention
        try:
            from flash_attn import flash_attn_varlen_func
            results["flash_attention"] = "available"
        except ImportError:
            results["flash_attention"] = "not_available"
        
        # Test triton
        try:
            import triton
            results["triton"] = "available"
        except ImportError:
            results["triton"] = "not_available"
        
        # Test transformers
        try:
            import transformers
            results["transformers"] = "available"
        except ImportError:
            results["transformers"] = "not_available"
        
        return {
            "status": "pass",
            "dependencies": results
        }
    
    def test_model_validation(self, model_names: List[str]) -> Dict[str, Any]:
        """Test model validation system."""
        logger.info("Testing model validation system...")
        
        results = {}
        for model_name in model_names:
            try:
                compatibility = check_model_compatibility(model_name)
                results[model_name] = {
                    "status": "pass" if compatibility["compatible"] else "fail",
                    "model_type": compatibility.get("model_type", "unknown"),
                    "model_size": compatibility.get("model_size", "unknown"),
                    "error": compatibility.get("error", None)
                }
            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        self.validation_results["model_validation"] = results
        return results
    
    def test_docker_build(self, dockerfile_path: str = "Dockerfile") -> Dict[str, Any]:
        """Test Docker build process."""
        logger.info("Testing Docker build process...")
        
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                return {
                    "status": "skip",
                    "reason": "Docker not available"
                }
            
            # Test build (dry run with small context)
            logger.info("Testing Docker build (this may take several minutes)...")
            build_result = subprocess.run([
                "docker", "build", 
                "--target", "builder",
                "--build-arg", "MODEL_NAME=Qwen/Qwen3-0.5B",  # Use small model for testing
                "-f", dockerfile_path,
                "-t", "nano-vllm-test",
                "."
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if build_result.returncode == 0:
                return {
                    "status": "pass",
                    "build_output": build_result.stdout[-1000:]  # Last 1000 chars
                }
            else:
                return {
                    "status": "fail",
                    "error": build_result.stderr[-2000:],  # Last 2000 chars of error
                    "stdout": build_result.stdout[-1000:]
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Docker build timed out after 10 minutes"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def test_handler_imports(self) -> Dict[str, Any]:
        """Test that handler imports work correctly."""
        logger.info("Testing handler imports...")
        
        try:
            # Test basic imports
            import torch
            import runpod
            
            # Test nano-vllm imports
            from nanovllm import LLM
            from nanovllm.sampling_params import SamplingParams
            
            # Test platform compatibility
            from nanovllm.utils.platform_compat import get_platform_compat
            compat = get_platform_compat()
            
            # Test model validation
            from nanovllm.model_validator import check_model_compatibility
            
            return {
                "status": "pass",
                "torch_version": torch.__version__,
                "platform": compat.system,
                "cuda_available": compat.is_cuda_available
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    def test_attention_fallback(self) -> Dict[str, Any]:
        """Test attention fallback mechanisms."""
        logger.info("Testing attention fallback mechanisms...")
        
        try:
            import torch
            from nanovllm.layers.attention import Attention
            
            # Create attention module
            num_heads = 8
            head_dim = 64
            scale = 1.0 / (head_dim ** 0.5)
            attention = Attention(num_heads, head_dim, scale, num_heads)
            
            # Test with CPU tensors
            batch_size = 2
            seq_len = 128
            q = torch.randn(batch_size, num_heads, seq_len, head_dim)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            
            # Mock context
            class MockContext:
                def __init__(self):
                    self.is_prefill = True
                    self.slot_mapping = torch.zeros(batch_size * seq_len, dtype=torch.long)
                    self.max_seqlen_q = seq_len
                    self.cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
                    self.max_seqlen_k = seq_len
                    self.cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32)
                    self.block_tables = None
                    self.context_lens = torch.tensor([seq_len], dtype=torch.int32)
            
            # Patch get_context
            import nanovllm.layers.attention
            original_get_context = nanovllm.layers.attention.get_context
            nanovllm.layers.attention.get_context = lambda: MockContext()
            
            try:
                # Test forward pass
                output = attention.forward(q, k, v)
                
                return {
                    "status": "pass",
                    "output_shape": list(output.shape),
                    "device": str(output.device),
                    "fallback_used": True
                }
            finally:
                # Restore original get_context
                nanovllm.layers.attention.get_context = original_get_context
                
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("=" * 60)
        report.append("NANO-VLLM DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Platform compatibility summary
        if "platform" in self.validation_results:
            platform = self.validation_results["platform"]
            report.append("PLATFORM COMPATIBILITY:")
            report.append(f"  System: {platform['platform_info']['system']}")
            report.append(f"  Machine: {platform['platform_info']['machine']}")
            report.append(f"  CUDA Available: {platform['platform_info']['cuda_available']}")
            if platform['platform_info']['cuda_version']:
                report.append(f"  CUDA Version: {platform['platform_info']['cuda_version']}")
            
            device_check = platform['device_check']
            report.append(f"  Device Check: {device_check['status']}")
            if device_check['status'] == 'pass':
                report.append(f"    Device: {device_check['device']}")
                report.append(f"    Dtype: {device_check['dtype']}")
            
            memory_check = platform['memory_check']
            report.append(f"  Memory Check: {memory_check['status']}")
            
            dependency_check = platform['dependency_check']
            report.append(f"  Dependencies: {dependency_check['status']}")
            for dep, status in dependency_check['dependencies'].items():
                report.append(f"    {dep}: {status}")
            report.append("")
        
        # Model validation summary
        if "model_validation" in self.validation_results:
            report.append("MODEL VALIDATION:")
            for model_name, result in self.validation_results["model_validation"].items():
                report.append(f"  {model_name}: {result['status']}")
                if result['status'] != 'pass':
                    report.append(f"    Error: {result.get('error', 'Unknown error')}")
            report.append("")
        
        # Overall status
        all_passed = all(
            result.get('status') == 'pass' 
            for category in self.validation_results.values() 
            for result in (category.values() if isinstance(category, dict) else [category])
        )
        
        report.append("OVERALL STATUS:")
        report.append(f"  Result: {'PASS' if all_passed else 'FAIL'}")
        report.append(f"  Tests Passed: {sum(1 for r in self._flatten_results() if r.get('status') == 'pass')}")
        report.append(f"  Tests Failed: {sum(1 for r in self._flatten_results() if r.get('status') == 'fail')}")
        report.append(f"  Tests Skipped: {sum(1 for r in self._flatten_results() if r.get('status') == 'skip')}")
        
        return "\n".join(report)
    
    def _flatten_results(self):
        """Flatten validation results for counting."""
        flattened = []
        for category in self.validation_results.values():
            if isinstance(category, dict):
                for result in category.values():
                    if isinstance(result, dict) and 'status' in result:
                        flattened.append(result)
            elif isinstance(category, dict) and 'status' in category:
                flattened.append(category)
        return flattened


async def main():
    """Main validation function."""
    logger.info("Starting nano-vLLM deployment validation...")
    
    validator = DeploymentValidator()
    
    # Test platform compatibility
    platform_results = validator.test_platform_compatibility()
    logger.info(f"Platform compatibility: {platform_results}")
    
    # Test model validation
    test_models = [
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1"
    ]
    model_results = validator.test_model_validation(test_models)
    logger.info(f"Model validation: {model_results}")
    
    # Test handler imports
    import_results = validator.test_handler_imports()
    logger.info(f"Handler imports: {import_results}")
    validator.validation_results["handler_imports"] = import_results
    
    # Test attention fallback
    attention_results = validator.test_attention_fallback()
    logger.info(f"Attention fallback: {attention_results}")
    validator.validation_results["attention_fallback"] = attention_results
    
    # Test Docker build (optional, can be slow)
    if "--skip-docker" not in sys.argv:
        docker_results = validator.test_docker_build()
        logger.info(f"Docker build: {docker_results}")
        validator.validation_results["docker_build"] = docker_results
    else:
        logger.info("Skipping Docker build test")
    
    # Generate and print report
    report = validator.generate_report()
    print("\n" + report)
    
    # Save report to file
    report_path = "deployment_validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())