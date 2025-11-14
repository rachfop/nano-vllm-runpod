import torch
import torch.distributed as dist
from torch import nn
from transformers import MistralConfig

from ..layers.activation import SiluAndMul
from ..layers.attention import Attention
from ..layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from ..layers.layernorm import RMSNorm
from ..layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from ..layers.rotary_embedding import get_rope
from .registry import register_model


@register_model("mistral")
class MistralForCausalLM(nn.Module):
    """Mistral model implementation for nano-vLLM."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)
        self.intermediate_size = config.intermediate_size
        self.rms_norm_eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.sliding_window = getattr(config, "sliding_window", None)

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList(
            [
                MistralDecoderLayer(self.config, layer_idx)
                for layer_idx in range(self.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.lm_head = ParallelLMHead(self.vocab_size, self.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute final logits."""
        return self.lm_head(self.norm(hidden_states))


class MistralDecoderLayer(nn.Module):
    """Mistral decoder layer."""

    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=getattr(config, "sliding_window", None),
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MistralAttention(nn.Module):
    """Mistral attention mechanism with sliding window support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        sliding_window: int | None = None,
    ):
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position,
            rope_theta,
            rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # KV cache will be set by ModelRunner
        self.k_cache = None
        self.v_cache = None

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply rotary embeddings
        q = self.rotary_emb(q, positions)
        k = self.rotary_emb(k, positions)
        
        # Attention computation (sliding window handled in attention layer)
        output = self.attn(q, k, v, positions)
        
        # Output projection
        output = self.o_proj(output)
        return output


class MistralMLP(nn.Module):
    """Mistral MLP layer."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate, up)
        x = self.down_proj(x)
        return x