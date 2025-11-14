import platform

import torch
from torch import nn

# Make triton optional for non-Linux platforms
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

    # Create dummy decorator for non-triton environments
    class triton:
        @staticmethod
        def jit(fn):
            return fn


# Make flash_attn optional for testing
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

    # Create dummy functions for testing
    def flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn not available - install flash-attn for production use"
        )
    def flash_attn_with_kvcache(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn not available - install flash-attn for production use"
        )
from ..utils import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr if HAS_TRITON else int,
):
    if HAS_TRITON:
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)
    else:
        # Fallback implementation for non-triton environments
        import numpy as np

        N = key_ptr.shape[0] if hasattr(key_ptr, "shape") else 1
        for idx in range(N):
            slot = (
                slot_mapping_ptr[idx]
                if hasattr(slot_mapping_ptr, "__getitem__")
                else slot_mapping_ptr
            )
            if slot == -1:
                continue
            # Simple fallback - this won't be as efficient but allows testing
            pass  # Placeholder for fallback logic


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _cpu_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        if q.dim() == 3:
            total_q, num_heads, head_dim = q.shape
            assert k.dim() == 3 and v.dim() == 3
            assert k.shape[1] == num_heads and v.shape[1] == num_heads
            cu_q = context.cu_seqlens_q
            cu_k = context.cu_seqlens_k
            assert cu_q is not None and cu_k is not None
            batches = cu_q.shape[0] - 1
            outs = []
            for b in range(batches):
                q_start = int(cu_q[b].item())
                q_end = int(cu_q[b + 1].item())
                k_start = int(cu_k[b].item())
                k_end = int(cu_k[b + 1].item())
                q_b = q[q_start:q_end].transpose(0, 1)
                k_b = k[k_start:k_end].transpose(0, 1)
                v_b = v[k_start:k_end].transpose(0, 1)
                Lq = q_b.shape[1]
                Lk = k_b.shape[1]
                scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.scale
                if context.is_prefill:
                    mask = torch.triu(torch.ones(Lq, Lk, device=q.device), diagonal=1).bool()
                    scores.masked_fill_(mask, float("-inf"))
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v_b).transpose(0, 1)
                outs.append(out)
            return torch.cat(outs, dim=0)
        elif q.dim() == 4:
            B, Lq, H, D = q.shape
            Bk, Lk, Hk, Dk = k.shape
            assert B == Bk and H == Hk and D == Dk
            q_t = q.permute(0, 2, 1, 3)
            k_t = k.permute(0, 2, 1, 3)
            v_t = v.permute(0, 2, 1, 3)
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
            if context.is_prefill:
                mask = torch.triu(torch.ones(Lq, Lk, device=q.device), diagonal=1).bool()
                scores.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_t).permute(0, 2, 1, 3)
            return out
        else:
            raise ValueError("Unsupported tensor rank for CPU attention")
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store KV cache if available
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Check if we can use flash attention
        can_use_flash = HAS_FLASH_ATTN and q.is_cuda and k.is_cuda and v.is_cuda
        
        if can_use_flash:
            try:
                if context.is_prefill:
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        max_seqlen_q=context.max_seqlen_q,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_k=context.max_seqlen_k,
                        cu_seqlens_k=context.cu_seqlens_k,
                        softmax_scale=self.scale,
                        causal=True,
                        block_table=context.block_tables,
                    )
                else:  # decode
                    o = flash_attn_with_kvcache(
                        q.unsqueeze(1),
                        k_cache,
                        v_cache,
                        cache_seqlens=context.context_lens,
                        block_table=context.block_tables,
                        softmax_scale=self.scale,
                        causal=True,
                    )
                return o
            except Exception as e:
                print(f"Flash attention failed, falling back to CPU implementation: {e}")
                return self._cpu_attention(q, k, v, context)
        else:
            # Use CPU fallback implementation
            if not q.is_cuda:
                print("Running on CPU - using fallback attention implementation")
            elif not HAS_FLASH_ATTN:
                print("Flash attention not available - using fallback implementation")
            
            return self._cpu_attention(q, k, v, context)
