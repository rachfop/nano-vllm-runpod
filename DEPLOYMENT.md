# nano-vLLM Runpod Edition - Deployment Guide

This guide covers deploying the nano-vLLM Runpod Edition fork to Runpod Hub.

## 🍴 About This Fork

This is a **fork** of the original [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) project, specifically adapted for production deployment on Runpod's serverless infrastructure.

### Key Modifications:
- ✅ Added Runpod serverless handler
- ✅ OpenAI-compatible API wrapper
- ✅ Production deployment configuration
- ✅ Docker containerization optimized for Runpod
- ✅ Comprehensive deployment tooling

## 🚀 Quick Deployment

### 1. Repository Setup
```bash
# Clone your fork
git clone https://github.com/your-username/nano-vllm-runpod.git
cd nano-vllm-runpod

# Verify configuration
python test_config.py
```

### 2. Docker Build & Test
```bash
# Build Docker image
docker build -t nano-vllm-runpod:latest .

# Test locally (optional)
docker run --gpus all -e MODEL_NAME="Qwen/Qwen3-8B" -p 8000:8000 nano-vllm-runpod:latest
```

### 3. Push to Registry
```bash
# Tag for your registry
docker tag nano-vllm-runpod:latest your-registry/nano-vllm-runpod:latest

# Push to registry
docker push your-registry/nano-vllm-runpod:latest
```

### 4. Runpod Hub Deployment

#### Option A: Manual Deployment
1. Log into [Runpod Console](https://www.runpod.io/console)
2. Navigate to "Deploy" → "Serverless"
3. Configure using settings from `.runpod/hub.json`
4. Set environment variables as needed

#### Option B: Hub Submission
1. Fork this repository
2. Update `.runpod/hub.json` with your registry URL
3. Submit to Runpod Hub for public deployment

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `"Qwen/Qwen3-8B"` | Hugging Face model ID |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs (1-8) |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | Batch size limit |
| `MAX_NUM_SEQS` | `512` | Concurrent sequences |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory usage (0.1-1.0) |
| `MAX_CONCURRENCY` | `30` | Max concurrent requests |
| `BASE_PATH` | `"/runpod-volume"` | Storage directory |

### Model Presets

The fork includes pre-configured settings for popular models:

```json
{
  "Qwen/Qwen3-1.7B": {
    "gpu_memory": "16GB",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8
  },
  "Qwen/Qwen3-8B": {
    "gpu_memory": "24GB",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9
  }
}
```

## 📊 Hardware Requirements

### GPU Recommendations
- **Small Models (1.7B)**: RTX 4090, A4000, or similar (16GB VRAM)
- **Medium Models (8B)**: RTX 4090, A5000, or similar (24GB VRAM)
- **Large Models**: Multiple GPUs with tensor parallelism

### Runpod GPU Types
Configure in `.runpod/hub.json`:
```json
"gpuIds": "ADA_80_PRO, AMPERE_80, AMPERE_48"
```

## 🔌 API Usage

### Basic Text Generation
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is artificial intelligence?",
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

### OpenAI-Compatible Format
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Explain quantum computing",
      "max_tokens": 150,
      "temperature": 0.8,
      "openai_route": true
    }
  }'
```

### Streaming Response
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Tell me a story",
      "max_tokens": 200,
      "temperature": 0.9,
      "stream": true
    }
  }'
```

## 🧪 Testing

### Local Testing
```bash
# Test configuration
python test_config.py

# Test with examples
python examples.py
```

### Deployment Testing
```bash
# Test endpoint health
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/health \
  -H "Authorization: Bearer YOUR_API_KEY"

# Test with sample input
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Hello, world!",
      "max_tokens": 10
    }
  }'
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Python Version Compatibility
**Problem**: nano-vLLM has issues with Python 3.12+
**Solution**: Use Python 3.10 or 3.11 in your Docker build

#### 2. GPU Memory Issues
**Problem**: Out of memory errors
**Solutions**:
- Reduce `GPU_MEMORY_UTILIZATION`
- Decrease `MAX_MODEL_LEN`
- Use a smaller model
- Increase GPU memory

#### 3. Model Loading Issues
**Problem**: Model fails to load
**Solutions**:
- Verify model ID is correct
- Check Hugging Face access permissions
- Ensure sufficient disk space
- Verify model compatibility with nano-vLLM

#### 4. Cold Start Performance
**Problem**: Slow initial requests
**Solutions**:
- Pre-download models in Docker image
- Use model caching
- Optimize container startup
- Consider keeping containers warm

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export RUNPOD_DEBUG=true
```

## 📈 Performance Optimization

### Throughput Tuning
```bash
# Increase concurrency
MAX_CONCURRENCY=50

# Optimize batching
MAX_NUM_BATCHED_TOKENS=32768
MAX_NUM_SEQS=1024

# Memory tuning
GPU_MEMORY_UTILIZATION=0.95
```

### Latency Optimization
```bash
# Reduce model length for faster inference
MAX_MODEL_LEN=2048

# Optimize tensor parallelism
TENSOR_PARALLEL_SIZE=2  # For multi-GPU
```

## 🔒 Security Considerations

- Use Runpod's secret management for API keys
- Implement rate limiting if needed
- Monitor usage and costs
- Secure your container registry
- Use HTTPS for all communications

## 📚 Additional Resources

- [Original nano-vLLM Documentation](https://github.com/GeeeekExplorer/nano-vllm)
- [Runpod Serverless Documentation](https://docs.runpod.io/serverless/)
- [Runpod Hub Guide](https://docs.runpod.io/hub/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🤝 Contributing

We welcome contributions! Please:
1. Fork this repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request with detailed description

## 📄 License

This fork maintains the MIT license of the original project. See [LICENSE](LICENSE) for details.

---

**🍴 Fork Information**: This is a production-ready fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) optimized for Runpod deployment.