import os
from dataclasses import dataclass

from huggingface_hub import snapshot_download
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    model_path: str | None = None
    tokenizer_source: str | None = None

    def __post_init__(self):
        self.model_path = self._ensure_model_path(self.model)
        self.tokenizer_source = self.model_path
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len

    def _ensure_model_path(self, model: str) -> str:
        """Return a local directory that contains the model weights."""
        if os.path.isdir(model):
            return model
        local_dir_root = os.getenv("MODEL_LOCAL_DIR") or os.getenv("MODEL_CACHE_DIR")
        snapshot_kwargs = {}
        if local_dir_root:
            local_dir = os.path.join(local_dir_root, model.replace("/", "_"))
            os.makedirs(local_dir, exist_ok=True)
            snapshot_kwargs["local_dir"] = local_dir
            snapshot_kwargs["local_dir_use_symlinks"] = False
        return snapshot_download(repo_id=model, **snapshot_kwargs)
