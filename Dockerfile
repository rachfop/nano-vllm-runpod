FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y python3-pip python3-dev git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up CUDA compatibility
RUN ldconfig /usr/local/cuda-12.1/compat/

# Set working directory
WORKDIR /app

# Configure CUDA path for builds that require it
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV PIP_ROOT_USER_ACTION=ignore

# Install Python dependencies first (for better caching)
COPY builder/requirements-core.txt /app/builder/requirements-core.txt
COPY builder/requirements.txt /app/builder/requirements.txt
COPY builder/requirements-optional.txt /app/builder/requirements-optional.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /app/builder/requirements-core.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-build-isolation --upgrade -r /app/builder/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-build-isolation --upgrade -r /app/builder/requirements-optional.txt

# Copy project files required for installation
COPY README.md LICENSE pyproject.toml /app/
COPY nanovllm /app/nanovllm
COPY handler.py /app/handler.py

# Install nano-vllm in development mode
RUN python3 -m pip install -e .

# Set environment variables
ARG MODEL_NAME="Qwen/Qwen3-8B"
ENV MODEL_NAME=${MODEL_NAME}
ENV BASE_PATH="/runpod-volume"
ENV HF_HOME="${BASE_PATH}/huggingface-cache/hub"
ENV HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets"
ENV PYTHONPATH="/app"

# Create cache directories
RUN mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE}

# Set up entry point
CMD ["python3", "/app/handler.py"]
