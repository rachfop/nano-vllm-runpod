import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import runpod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.model_validator import check_model_compatibility, validate_deployment_config
from nanovllm.utils.platform_compat import get_platform_compat, setup_platform_specific_logging


class NanoVLLMEngine:
    """Wrapper for nano-vllm to provide OpenAI-compatible interface"""

    def __init__(self):
        # Setup platform-specific logging
        setup_platform_specific_logging()
        
        # Get platform compatibility
        self.platform_compat = get_platform_compat()
        logger.info("Platform compatibility initialized")
        
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
        self.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        self.base_path = os.getenv("BASE_PATH", "/runpod-volume")
        self.max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
        self.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

        # Get platform-specific settings
        platform_settings = self.platform_compat.get_recommended_settings(self.model_name)
        logger.info(f"Platform settings: {platform_settings}")
        
        # Validate model compatibility
        logger.info(f"Checking model compatibility for: {self.model_name}")
        compatibility_check = check_model_compatibility(self.model_name)
        
        if not compatibility_check["compatible"]:
            logger.error(f"Model {self.model_name} is not compatible: {compatibility_check['error']}")
            logger.info(f"Supported model types: {', '.join(compatibility_check.get('supported_types', []))}")
            raise ValueError(f"Model {self.model_name} is not compatible with nano-vLLM")
        
        # Validate model size against platform limits
        model_size_bytes = compatibility_check.get("model_size_bytes", 0)
        if not self.platform_compat.validate_model_size(model_size_bytes):
            logger.error(f"Model {self.model_name} exceeds platform memory limits")
            raise ValueError(f"Model too large for current platform")
        
        logger.info(f"✅ Model {self.model_name} is compatible")
        logger.info(f"Model type: {compatibility_check['model_type']}")
        logger.info(f"Model size: {compatibility_check['model_size']}")
        logger.info(f"Max context: {compatibility_check['max_context_length']}")

        # Apply platform-specific limits
        self.max_model_len = min(self.max_model_len, platform_settings["max_sequence_length"])
        logger.info(f"Adjusted max model length: {self.max_model_len}")

        # Initialize nano-vllm engine with platform optimizations
        try:
            # Use platform-specific device and dtype
            device = self.platform_compat.get_optimal_device()
            dtype = self.platform_compat.get_optimal_dtype()
            
            logger.info(f"Using device: {device}, dtype: {dtype}")
            
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=int(os.getenv("MAX_NUM_BATCHED_TOKENS", "16384")),
                max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "512")),
                gpu_memory_utilization=self.gpu_memory_utilization,
                device=device.type,
                dtype=dtype,
            )
            logger.info("✅ nano-vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize nano-vLLM engine: {str(e)}")
            logger.info("💡 Tips: Check GPU memory, model path, and CUDA compatibility")
            logger.info(f"Platform info: {self.platform_compat.config}")
            raise

        # Set concurrency based on platform capabilities
        platform_max_concurrency = platform_settings["max_batch_size"]
        env_concurrency = int(os.getenv("MAX_CONCURRENCY", "30"))
        self.max_concurrency = min(env_concurrency, platform_max_concurrency)
        logger.info(f"Max concurrency set to: {self.max_concurrency} (platform limit: {platform_max_concurrency})")

    def create_sampling_params(self, params: Dict[str, Any]) -> SamplingParams:
        """Convert OpenAI-style parameters to nano-vllm SamplingParams"""
        return SamplingParams(
            temperature=params.get("temperature", 1.0),
            max_tokens=params.get("max_tokens", 64),
            ignore_eos=params.get("ignore_eos", False),
        )

    async def generate(
        self, job_input: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate text using nano-vllm"""
        try:
            # Extract parameters
            prompt = job_input.get("prompt", "")
            openai_route = job_input.get("openai_route", False)
            stream = job_input.get("stream", False)

            # Create sampling parameters
            sampling_params = self.create_sampling_params(job_input)

            # Generate with nano-vllm
            # Note: nano-vllm's generate method returns a list, we need to adapt it
            outputs = self.llm.generate(prompt, sampling_params)

            if stream:
                # For streaming, yield tokens incrementally
                for i, output in enumerate(outputs):
                    if hasattr(output, "text"):
                        yield {
                            "text": output.text,
                            "finish_reason": getattr(output, "finish_reason", "stop"),
                            "index": i,
                        }
            else:
                # For non-streaming, return complete response
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    response = {
                        "text": getattr(output, "text", str(output)),
                        "finish_reason": getattr(output, "finish_reason", "stop"),
                        "model": self.model_name,
                    }

                    # Add OpenAI-compatible format if requested
                    if openai_route:
                        response = {
                            "id": f"nano-vllm-{hash(prompt) % 10000}",
                            "object": "text_completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": self.model_name,
                            "choices": [
                                {
                                    "text": response["text"],
                                    "index": 0,
                                    "finish_reason": response["finish_reason"],
                                }
                            ],
                        }

                    yield response
                else:
                    yield {"error": "No output generated"}

        except Exception as e:
            yield {"error": str(e)}


# Initialize the engine
engine = NanoVLLMEngine()


async def handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """Main Runpod handler function"""
    job_input = job.get("input", {})

    # Validate input
    if not job_input.get("prompt"):
        yield {"error": "Missing 'prompt' in input"}
        return

    # Generate response
    async for result in engine.generate(job_input):
        yield result


# Start Runpod serverless
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
