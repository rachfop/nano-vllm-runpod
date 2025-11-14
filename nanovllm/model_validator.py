"""Model compatibility checker and validator for nano-vLLM."""

import logging
from typing import Dict, List, Optional, Tuple
from transformers import AutoConfig

from .models.registry import available_model_types, get_model_class

logger = logging.getLogger(__name__)

# Supported model families and their characteristics
SUPPORTED_MODELS = {
    "qwen3": {
        "description": "Qwen3 models (Alibaba)",
        "examples": ["Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.5B"],
        "features": ["rope", "swiglu", "rmsnorm"],
        "max_context": 32768,
    },
    "llama": {
        "description": "Llama and Llama 2/3 models (Meta)",
        "examples": ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B"],
        "features": ["rope", "swiglu", "rmsnorm"],
        "max_context": 8192,
    },
    "mistral": {
        "description": "Mistral and Mixtral models",
        "examples": ["mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"],
        "features": ["rope", "swiglu", "rmsnorm", "sliding_window"],
        "max_context": 8192,
    },
}

# Hardware requirements mapping
HARDWARE_REQUIREMENTS = {
    "1.7B": {"gpu_memory_gb": 8, "recommended_gpu": "RTX 4070, A4000"},
    "3B": {"gpu_memory_gb": 12, "recommended_gpu": "RTX 4080, A4500"},
    "7B": {"gpu_memory_gb": 16, "recommended_gpu": "RTX 4090, A5000"},
    "8B": {"gpu_memory_gb": 24, "recommended_gpu": "RTX 4090, A5000"},
    "13B": {"gpu_memory_gb": 32, "recommended_gpu": "A6000, A100-40GB"},
    "30B": {"gpu_memory_gb": 64, "recommended_gpu": "A100-80GB, H100"},
    "70B": {"gpu_memory_gb": 128, "recommended_gpu": "2x A100-80GB, H100"},
}


def check_model_compatibility(model_name: str) -> Dict[str, any]:
    """
    Check if a model is compatible with nano-vLLM.
    
    Args:
        model_name: HuggingFace model name or path
        
    Returns:
        Dict with compatibility status and recommendations
    """
    try:
        # Get model config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)
        
        if model_type is None:
            return {
                "compatible": False,
                "error": "Could not determine model type from config",
                "recommendation": "Check if the model is a valid HuggingFace model",
            }
        
        # Check if model type is supported
        available_types = available_model_types()
        if model_type not in available_types:
            supported_list = list(SUPPORTED_MODELS.keys())
            return {
                "compatible": False,
                "error": f"Model type '{model_type}' is not supported",
                "supported_types": supported_list,
                "recommendation": f"Use one of these supported models: {', '.join(supported_list)}",
            }
        
        # Get model size estimate
        model_size = estimate_model_size(config)
        hardware_req = get_hardware_requirements(model_size)
        
        # Check context length
        max_model_len = getattr(config, "max_position_embeddings", 2048)
        
        return {
            "compatible": True,
            "model_type": model_type,
            "model_size": model_size,
            "max_context_length": max_model_len,
            "hardware_requirements": hardware_req,
            "features": SUPPORTED_MODELS.get(model_type, {}).get("features", []),
            "recommendation": f"Model is compatible. Recommended GPU: {hardware_req['recommended_gpu']}",
        }
        
    except Exception as e:
        return {
            "compatible": False,
            "error": f"Failed to check model compatibility: {str(e)}",
            "recommendation": "Ensure the model is accessible and properly configured",
        }


def estimate_model_size(config) -> str:
    """Estimate model size based on config parameters."""
    hidden_size = getattr(config, "hidden_size", 0)
    num_layers = getattr(config, "num_hidden_layers", 0)
    vocab_size = getattr(config, "vocab_size", 0)
    
    # Rough parameter count estimation
    params_billion = (hidden_size * hidden_size * num_layers * 12 + vocab_size * hidden_size) / 1e9
    
    if params_billion < 3:
        return "1.7B"
    elif params_billion < 8:
        return "7B"
    elif params_billion < 15:
        return "13B"
    elif params_billion < 35:
        return "30B"
    else:
        return "70B"


def get_hardware_requirements(model_size: str) -> Dict[str, any]:
    """Get hardware requirements for a given model size."""
    return HARDWARE_REQUIREMENTS.get(model_size, {
        "gpu_memory_gb": 24,
        "recommended_gpu": "RTX 4090, A5000 (check specific model requirements)",
    })


def validate_deployment_config(model_name: str, gpu_memory_gb: int, max_model_len: int) -> Dict[str, any]:
    """
    Validate deployment configuration.
    
    Args:
        model_name: Model name
        gpu_memory_gb: Available GPU memory in GB
        max_model_len: Maximum sequence length
        
    Returns:
        Validation results with recommendations
    """
    compatibility = check_model_compatibility(model_name)
    
    if not compatibility["compatible"]:
        return compatibility
    
    model_size = compatibility["model_size"]
    required_memory = HARDWARE_REQUIREMENTS.get(model_size, {}).get("gpu_memory_gb", 24)
    
    warnings = []
    recommendations = []
    
    # Check GPU memory
    if gpu_memory_gb < required_memory:
        warnings.append(f"Insufficient GPU memory: {gpu_memory_gb}GB < {required_memory}GB required")
        recommendations.append(f"Use a GPU with at least {required_memory}GB memory")
    
    # Check context length
    model_max_context = compatibility.get("max_context_length", 2048)
    if max_model_len > model_max_context:
        warnings.append(f"Requested context length {max_model_len} > model's max {model_max_context}")
        recommendations.append(f"Reduce MAX_MODEL_LEN to {model_max_context}")
    
    return {
        "compatible": len(warnings) == 0,
        "warnings": warnings,
        "recommendations": recommendations,
        "model_info": compatibility,
    }


def get_supported_models() -> Dict[str, List[str]]:
    """Get list of supported models with examples."""
    return {
        model_type: info["examples"] 
        for model_type, info in SUPPORTED_MODELS.items()
    }


def print_model_compatibility_report(model_name: str):
    """Print a detailed compatibility report for a model."""
    result = check_model_compatibility(model_name)
    
    print(f"\n{'='*60}")
    print(f"Model Compatibility Report: {model_name}")
    print(f"{'='*60}")
    
    if result["compatible"]:
        print(f"✅ COMPATIBLE")
        print(f"Model Type: {result['model_type']}")
        print(f"Model Size: {result['model_size']}")
        print(f"Max Context: {result['max_context_length']}")
        print(f"Features: {', '.join(result['features'])}")
        print(f"Hardware: {result['hardware_requirements']['recommended_gpu']}")
    else:
        print(f"❌ NOT COMPATIBLE")
        print(f"Error: {result['error']}")
        if 'supported_types' in result:
            print(f"Supported types: {', '.join(result['supported_types'])}")
    
    if 'recommendation' in result:
        print(f"Recommendation: {result['recommendation']}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the compatibility checker
    test_models = [
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-2-7b-hf", 
        "mistralai/Mistral-7B-v0.1",
        "microsoft/DialoGPT-medium",  # Unsupported
    ]
    
    for model in test_models:
        try:
            print_model_compatibility_report(model)
        except Exception as e:
            print(f"Error checking {model}: {e}")