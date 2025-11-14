"""Platform compatibility utilities for cross-platform deployment."""

import platform
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PlatformCompat:
    """Platform compatibility manager for handling different deployment environments."""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.is_cuda_available = torch.cuda.is_available()
        self.cuda_version = self._get_cuda_version() if self.is_cuda_available else None
        
        # Platform-specific configurations
        self.config = self._get_platform_config()
        
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version if available."""
        try:
            if torch.cuda.is_available():
                return torch.version.cuda
        except Exception as e:
            logger.warning(f"Could not determine CUDA version: {e}")
        return None
    
    def _get_platform_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration."""
        base_config = {
            "max_batch_size": 32,
            "max_sequence_length": 4096,
            "use_flash_attention": True,
            "use_triton": True,
            "dtype": "float16",
            "device": "cuda" if self.is_cuda_available else "cpu",
        }
        
        # Platform-specific adjustments
        if self.system == "Darwin":  # macOS
            config = base_config.copy()
            config.update({
                "use_flash_attention": False,  # flash-attn not supported on macOS
                "use_triton": False,  # triton not supported on macOS
                "device": "mps" if torch.backends.mps.is_available() else "cpu",
                "dtype": "float32",  # MPS works better with float32
                "max_batch_size": 16,  # Conservative for macOS
            })
            return config
            
        elif self.system == "Linux":
            config = base_config.copy()
            if not self.is_cuda_available:
                config.update({
                    "use_flash_attention": False,
                    "use_triton": False,
                    "device": "cpu",
                    "dtype": "float32",
                    "max_batch_size": 8,  # Conservative for CPU
                })
            return config
            
        elif self.system == "Windows":
            config = base_config.copy()
            config.update({
                "use_flash_attention": False,  # flash-attn often problematic on Windows
                "use_triton": False,  # triton not well supported on Windows
                "max_batch_size": 16,  # Conservative for Windows
            })
            if not self.is_cuda_available:
                config.update({
                    "device": "cpu",
                    "dtype": "float32",
                    "max_batch_size": 8,
                })
            return config
            
        else:
            # Fallback for other platforms
            config = base_config.copy()
            config.update({
                "use_flash_attention": False,
                "use_triton": False,
                "device": "cpu",
                "dtype": "float32",
                "max_batch_size": 8,
            })
            return config
    
    def get_optimal_device(self) -> torch.device:
        """Get the optimal device for the current platform."""
        if self.config["device"] == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.config["device"] == "cuda" and self.is_cuda_available:
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get the optimal dtype for the current platform."""
        dtype_str = self.config["dtype"]
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "float32":
            return torch.float32
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def can_use_flash_attention(self) -> bool:
        """Check if flash attention can be used on this platform."""
        return self.config["use_flash_attention"] and self.is_cuda_available
    
    def can_use_triton(self) -> bool:
        """Check if triton can be used on this platform."""
        return self.config["use_triton"] and self.system == "Linux" and self.is_cuda_available
    
    def get_memory_limits(self) -> Dict[str, int]:
        """Get memory limits for the current platform."""
        if self.is_cuda_available:
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return {
                "total_memory": gpu_memory,
                "max_model_memory": int(gpu_memory * 0.8),  # 80% for model
                "max_cache_memory": int(gpu_memory * 0.15),  # 15% for cache
                "reserved_memory": int(gpu_memory * 0.05),   # 5% reserved
            }
        else:
            # Conservative CPU memory limits
            return {
                "total_memory": 16 * 1024**3,  # 16GB default
                "max_model_memory": 8 * 1024**3,   # 8GB for model
                "max_cache_memory": 4 * 1024**3,   # 4GB for cache
                "reserved_memory": 4 * 1024**3,   # 4GB reserved
            }
    
    def validate_model_size(self, model_size_bytes: int) -> bool:
        """Validate if model size is suitable for current platform."""
        memory_limits = self.get_memory_limits()
        max_model_memory = memory_limits["max_model_memory"]
        
        if model_size_bytes > max_model_memory:
            logger.warning(
                f"Model size ({model_size_bytes / 1024**3:.1f}GB) exceeds "
                f"platform limit ({max_model_memory / 1024**3:.1f}GB)"
            )
            return False
        return True
    
    def get_recommended_settings(self, model_name: str) -> Dict[str, Any]:
        """Get recommended settings for a specific model on this platform."""
        settings = self.config.copy()
        
        # Model-specific adjustments
        if "7b" in model_name.lower() or "8b" in model_name.lower():
            if not self.is_cuda_available:
                settings["max_batch_size"] = min(settings["max_batch_size"], 4)
                settings["max_sequence_length"] = min(settings["max_sequence_length"], 2048)
        elif "13b" in model_name.lower() or "30b" in model_name.lower():
            if not self.is_cuda_available:
                settings["max_batch_size"] = 1
                settings["max_sequence_length"] = min(settings["max_sequence_length"], 1024)
        elif "70b" in model_name.lower() or "65b" in model_name.lower():
            if not self.is_cuda_available:
                logger.error(f"Model {model_name} too large for CPU inference")
                raise ValueError(f"Model {model_name} requires GPU acceleration")
        
        return settings
    
    def log_platform_info(self):
        """Log platform information for debugging."""
        logger.info(f"Platform: {self.system} {self.machine}")
        logger.info(f"CUDA available: {self.is_cuda_available}")
        if self.cuda_version:
            logger.info(f"CUDA version: {self.cuda_version}")
        logger.info(f"Optimal device: {self.get_optimal_device()}")
        logger.info(f"Optimal dtype: {self.get_optimal_dtype()}")
        logger.info(f"Flash attention: {self.can_use_flash_attention()}")
        logger.info(f"Triton: {self.can_use_triton()}")


# Global platform compatibility instance
_platform_compat = None

def get_platform_compat() -> PlatformCompat:
    """Get the global platform compatibility instance."""
    global _platform_compat
    if _platform_compat is None:
        _platform_compat = PlatformCompat()
    return _platform_compat


def setup_platform_specific_logging():
    """Setup logging with platform-specific configurations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nano_vllm_platform.log')
        ]
    )
    
    # Log platform info on startup
    compat = get_platform_compat()
    compat.log_platform_info()