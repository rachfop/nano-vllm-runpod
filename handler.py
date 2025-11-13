import os
import asyncio
import json
from typing import AsyncGenerator, Dict, Any
import runpod
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


class NanoVLLMEngine:
    """Wrapper for nano-vllm to provide OpenAI-compatible interface"""
    
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
        self.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        self.base_path = os.getenv("BASE_PATH", "/runpod-volume")
        
        # Initialize nano-vllm engine
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
            max_num_batched_tokens=int(os.getenv("MAX_NUM_BATCHED_TOKENS", "16384")),
            max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "512")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
        )
        
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", "30"))
    
    def create_sampling_params(self, params: Dict[str, Any]) -> SamplingParams:
        """Convert OpenAI-style parameters to nano-vllm SamplingParams"""
        return SamplingParams(
            temperature=params.get("temperature", 1.0),
            max_tokens=params.get("max_tokens", 64),
            ignore_eos=params.get("ignore_eos", False)
        )
    
    async def generate(self, job_input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
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
                    if hasattr(output, 'text'):
                        yield {
                            "text": output.text,
                            "finish_reason": getattr(output, 'finish_reason', 'stop'),
                            "index": i
                        }
            else:
                # For non-streaming, return complete response
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    response = {
                        "text": getattr(output, 'text', str(output)),
                        "finish_reason": getattr(output, 'finish_reason', 'stop'),
                        "model": self.model_name
                    }
                    
                    # Add OpenAI-compatible format if requested
                    if openai_route:
                        response = {
                            "id": f"nano-vllm-{hash(prompt) % 10000}",
                            "object": "text_completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": self.model_name,
                            "choices": [{
                                "text": response["text"],
                                "index": 0,
                                "finish_reason": response["finish_reason"]
                            }]
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
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda x: engine.max_concurrency,
    "return_aggregate_stream": True,
})