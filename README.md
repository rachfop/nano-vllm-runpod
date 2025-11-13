# nano-vLLM Runpod Edition

A production-ready fork of nano-vLLM optimized for Runpod serverless deployment.

## 🚀 Overview

This fork adapts the excellent [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) project for production deployment on Runpod Hub, providing:

- **Serverless Architecture**: Optimized for Runpod's serverless GPU infrastructure
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Production Ready**: Built-in error handling, monitoring, and scaling
- **Easy Deployment**: One-click deployment via Runpod Hub

## 📋 Original Project

This is a fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) - a lightweight vLLM implementation built from scratch for educational purposes.

**Key Changes from Original:**
- Added Runpod serverless handler
- OpenAI-compatible API wrapper
- Production deployment configuration
- Docker containerization
- Hub deployment specification

## 🔧 Quick Start

### Local Development
```bash
# Clone the fork
git clone https://github.com/your-username/nano-vllm-runpod.git
cd nano-vllm-runpod

# Install dependencies
pip install -e .

# Test configuration
python test_config.py
```

### Runpod Deployment
1. Build Docker image: `docker build -t nano-vllm-runpod .`
2. Push to container registry
3. Deploy via Runpod Hub using `.runpod/hub.json`

## 📊 Performance

- **Model Support**: Qwen3, and other compatible models
- **GPU Requirements**: 16GB+ VRAM recommended
- **Throughput**: Configurable via environment variables
- **Latency**: Optimized for serverless cold starts

## 🛠️ Configuration

Key environment variables:
```bash
MODEL_NAME="Qwen/Qwen3-8B"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9
MAX_CONCURRENCY=30
```

## 📚 API Usage

### Basic Text Generation
```json
{
  "input": {
    "prompt": "What is AI?",
    "max_tokens": 100,
    "temperature": 0.7
  }
}
```

### OpenAI-Compatible
```json
{
  "input": {
    "prompt": "Explain quantum computing",
    "openai_route": true
  }
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork this repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This fork maintains the same MIT license as the original nano-vLLM project.

## 🙏 Acknowledgments

- Original nano-vLLM by [GeeeekExplorer](https://github.com/GeeeekExplorer)
- Runpod for the serverless infrastructure
- Hugging Face for model hosting