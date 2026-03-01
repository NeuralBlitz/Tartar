“””
Claude-Inspired Transformer Architecture
Based on publicly known details of Anthropic’s Claude:

- Transformer decoder architecture
- Multi-head attention with rotary positional embeddings (RoPE)
- SwiGLU activation functions
- RMSNorm (instead of LayerNorm)
- Constitutional AI training hooks
- KV-cache for efficient inference
  “””

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# ─────────────────────────────────────────────

# CONFIG

# ─────────────────────────────────────────────

@dataclass
class ClaudeConfig:
“””
Approximate config inspired by Claude’s known architecture.
Scaled down for practical use - scale up for production.
“””
vocab_size: int = 32000
hidden_dim: int = 4096          # Model dimension
num_layers: int = 32            # Transformer blocks
num_heads: int = 32             # Attention heads
num_kv_heads: int = 8           # GQA: fewer KV heads (like Claude)
head_dim: int = 128             # Dimension per head
ffn_dim: int = 11008            # FFN intermediate size (~2.7x hidden)
max_seq_len: int = 100000       # Claude’s long context window
dropout: float = 0.0            # No dropout at inference
rope_theta: float = 500000.0    # RoPE base frequency (long context)
norm_eps: float = 1e-5
tie_embeddings: bool = True     # Tie input/output embeddings

```
# Constitutional AI hooks
use_constitutional_filter: bool = True
harmlessness_weight: float = 0.5
helpfulness_weight: float = 0.5
```

# ─────────────────────────────────────────────

# RMS NORM (Claude uses this instead of LayerNorm)

# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
“”“Root Mean Square Layer Normalization - more efficient than LayerNorm”””

```
def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute RMS
    rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
    return x / rms * self.weight
```

# ─────────────────────────────────────────────

# ROTARY POSITIONAL EMBEDDINGS (RoPE)

# ─────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
“””
Rotary Position Embedding - encodes position via rotation in complex space.
Claude uses an extended RoPE for very long context windows.
“””

```
def __init__(self, dim: int, max_seq_len: int = 100000, theta: float = 500000.0):
    super().__init__()
    self.dim = dim
    self.max_seq_len = max_seq_len
    self.theta = theta

    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq)

    # Precompute cos/sin cache
    self._build_cache(max_seq_len)

def _build_cache(self, seq_len: int):
    t = torch.arange(seq_len, device=self.inv_freq.device).float()
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    self.register_buffer("cos_cache", emb.cos())
    self.register_buffer("sin_cache", emb.sin())

def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (self.rotate_half(q) * sin)
    k_rot = (k * cos) + (self.rotate_half(k) * sin)

    return q_rot, k_rot
```

# ─────────────────────────────────────────────

# GROUPED QUERY ATTENTION (GQA)

# ─────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
“””
Grouped Query Attention - Claude uses fewer KV heads than Q heads.
This reduces memory bandwidth without sacrificing much quality.
Also implements KV-cache for efficient autoregressive generation.
“””

```
def __init__(self, config: ClaudeConfig):
    super().__init__()
    self.num_heads = config.num_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.hidden_dim = config.hidden_dim
    self.groups = config.num_heads // config.num_kv_heads

    # Projections
    self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * config.head_dim, bias=False)
    self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * config.head_dim, bias=False)
    self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_dim, bias=False)

    self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    self.scale = config.head_dim ** -0.5

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    B, T, _ = x.shape

    # Project Q, K, V
    q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = self.rope(q, k, T)

    # KV Cache: append new K, V
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)

    new_kv_cache = (k, v)

    # Expand KV heads to match Q heads (GQA)
    k = k.repeat_interleave(self.groups, dim=1)
    v = v.repeat_interleave(self.groups, dim=1)

    # Scaled dot-product attention
    # Use Flash Attention if available
    if hasattr(F, 'scaled_dot_product_attention'):
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=(mask is None)
        )
    else:
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights + mask
        else:
            # Causal mask
            causal_mask = torch.triu(
                torch.full((T, k.shape[2]), float('-inf'), device=x.device),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

    # Reshape and project output
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
    out = self.o_proj(attn_out)

    return out, new_kv_cache
```

# ─────────────────────────────────────────────

# SwiGLU FEED-FORWARD NETWORK

# ─────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
“””
SwiGLU Feed-Forward Network.
Claude (and most modern LLMs) use SwiGLU instead of ReLU/GELU.
FFN(x) = (Swish(xW1) ⊙ xW3) W2
“””

```
def __init__(self, config: ClaudeConfig):
    super().__init__()
    self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
    self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
    self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    # SwiGLU: element-wise multiply gated activation
    gate = F.silu(self.gate_proj(x))  # Swish activation
    up = self.up_proj(x)
    return self.down_proj(gate * up)
```

# ─────────────────────────────────────────────

# TRANSFORMER BLOCK

# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
“””
Single transformer decoder block:
Pre-norm → Attention → Residual → Pre-norm → FFN → Residual
“””

```
def __init__(self, config: ClaudeConfig):
    super().__init__()
    self.attn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
    self.attn = GroupedQueryAttention(config)
    self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
    self.ffn = SwiGLUFFN(config)

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple] = None,
) -> Tuple[torch.Tensor, Tuple]:

    # Attention with pre-norm + residual
    normed = self.attn_norm(x)
    attn_out, new_kv_cache = self.attn(normed, mask=mask, kv_cache=kv_cache)
    x = x + attn_out

    # FFN with pre-norm + residual
    x = x + self.ffn(self.ffn_norm(x))

    return x, new_kv_cache
```

# ─────────────────────────────────────────────

# CONSTITUTIONAL AI FILTER

# ─────────────────────────────────────────────

class ConstitutionalFilter(nn.Module):
“””
Simplified Constitutional AI scoring layer.
In real Claude, this is baked into RLHF/RLAIF training.
Here modeled as a learned scoring head over hidden states.

```
Scores outputs on:
- Helpfulness
- Harmlessness
- Honesty
"""

def __init__(self, hidden_dim: int):
    super().__init__()
    self.scorer = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.SiLU(),
        nn.Linear(512, 3)  # [helpfulness, harmlessness, honesty]
    )
    self.softmax = nn.Softmax(dim=-1)

def forward(self, hidden_states: torch.Tensor) -> dict:
    # Pool over sequence
    pooled = hidden_states.mean(dim=1)
    scores = self.scorer(pooled)

    return {
        "helpfulness": scores[:, 0],
        "harmlessness": scores[:, 1],
        "honesty": scores[:, 2],
        "constitutional_score": scores.mean(dim=-1)
    }
```

# ─────────────────────────────────────────────

# MAIN MODEL

# ─────────────────────────────────────────────

class ClaudeModel(nn.Module):
“””
Full Claude-inspired transformer model.

```
Architecture:
- Token embeddings
- N x TransformerBlock (RMSNorm + GQA + SwiGLU)
- Final RMSNorm
- LM Head (tied to embeddings)
- Constitutional AI scoring head
"""

def __init__(self, config: ClaudeConfig):
    super().__init__()
    self.config = config

    # Token embeddings
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

    # Transformer layers
    self.layers = nn.ModuleList([
        TransformerBlock(config) for _ in range(config.num_layers)
    ])

    # Final norm
    self.norm = RMSNorm(config.hidden_dim, config.norm_eps)

    # Language model head
    self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    # Tie embeddings (weight sharing)
    if config.tie_embeddings:
        self.lm_head.weight = self.embed_tokens.weight

    # Constitutional AI filter
    if config.use_constitutional_filter:
        self.constitutional_filter = ConstitutionalFilter(config.hidden_dim)

    # Initialize weights
    self.apply(self._init_weights)

    print(f"Claude-inspired model initialized")
    print(f"Parameters: {self.count_parameters():,}")

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

def forward(
    self,
    input_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kv_caches: Optional[list] = None,
    return_constitutional_scores: bool = False,
) -> dict:

    B, T = input_ids.shape

    # Embed tokens
    x = self.embed_tokens(input_ids)

    # Run through transformer layers
    new_kv_caches = []
    for i, layer in enumerate(self.layers):
        cache = kv_caches[i] if kv_caches is not None else None
        x, new_cache = layer(x, mask=mask, kv_cache=cache)
        new_kv_caches.append(new_cache)

    # Final norm
    x = self.norm(x)

    # LM head → logits
    logits = self.lm_head(x)

    output = {
        "logits": logits,
        "hidden_states": x,
        "kv_caches": new_kv_caches,
    }

    # Constitutional scoring
    if return_constitutional_scores and self.config.use_constitutional_filter:
        output["constitutional_scores"] = self.constitutional_filter(x)

    return output

@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """
    Autoregressive generation with:
    - Temperature sampling
    - Top-p (nucleus) sampling
    - Top-k sampling
    - Repetition penalty
    - KV-cache for efficiency
    """
    self.eval()
    device = input_ids.device
    B = input_ids.shape[0]

    generated = input_ids.clone()
    kv_caches = None

    for step in range(max_new_tokens):
        # Use only new tokens if we have cache
        if kv_caches is not None:
            curr_input = generated[:, -1:]
        else:
            curr_input = generated

        # Forward pass
        out = self.forward(curr_input, kv_caches=kv_caches)
        logits = out["logits"][:, -1, :]  # Last token logits
        kv_caches = out["kv_caches"]

        # Repetition penalty
        if repetition_penalty != 1.0:
            for b in range(B):
                for token_id in set(generated[b].tolist()):
                    if logits[b, token_id] < 0:
                        logits[b, token_id] *= repetition_penalty
                    else:
                        logits[b, token_id] /= repetition_penalty

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
            logits[logits < top_k_vals[:, -1:]] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_indices_to_remove[:, 0] = False  # Keep at least one token
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        # Simple EOS check (token 2 = </s> in most tokenizers)
        if (next_token == 2).all():
            break

    return generated
```

# ─────────────────────────────────────────────

# TRAINING SETUP (RLHF-style)

# ─────────────────────────────────────────────

class RewardModel(nn.Module):
“””
Reward model for RLHF training (how Claude learns human preferences).
Takes a sequence and outputs a scalar reward.
“””

```
def __init__(self, config: ClaudeConfig):
    super().__init__()
    # Reuse the base model architecture
    self.base = ClaudeModel(config)
    # Replace LM head with reward head
    self.reward_head = nn.Linear(config.hidden_dim, 1, bias=False)

def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    out = self.base(input_ids)
    hidden = out["hidden_states"]
    # Pool over sequence and compute reward
    pooled = hidden[:, -1, :]  # Use last token (like GPT reward models)
    reward = self.reward_head(pooled)
    return reward.squeeze(-1)
```

class PPOTrainer:
“””
Proximal Policy Optimization for RLHF.
This is how Claude is trained to be helpful, harmless, and honest.
Simplified implementation showing the core loop.
“””

```
def __init__(
    self,
    policy_model: ClaudeModel,
    reward_model: RewardModel,
    ref_model: ClaudeModel,  # Frozen reference policy
    lr: float = 1e-5,
    kl_coeff: float = 0.1,   # KL penalty to prevent policy collapse
    clip_eps: float = 0.2,
):
    self.policy = policy_model
    self.reward = reward_model
    self.ref = ref_model
    self.kl_coeff = kl_coeff
    self.clip_eps = clip_eps
    self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    # Freeze reference model
    for param in self.ref.parameters():
        param.requires_grad = False

def compute_kl_divergence(
    self,
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor
) -> torch.Tensor:
    """KL(policy || reference) - keeps policy close to base model"""
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    kl = (ref_probs * (ref_probs.log() - policy_log_probs)).sum(-1)
    return kl.mean()

def ppo_step(
    self,
    input_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> dict:
    """Single PPO update step"""

    # Forward pass through policy
    policy_out = self.policy(input_ids)
    policy_logits = policy_out["logits"]

    # Forward pass through reference (no grad)
    with torch.no_grad():
        ref_out = self.ref(input_ids)
        ref_logits = ref_out["logits"]

    # Compute log probs
    log_probs = F.log_softmax(policy_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    # PPO clipped objective
    ratio = (token_log_probs - old_log_probs).exp()
    clipped_ratio = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # KL penalty
    kl_loss = self.compute_kl_divergence(policy_logits, ref_logits)

    # Total loss
    total_loss = policy_loss + self.kl_coeff * kl_loss

    # Optimize
    self.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
    self.optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
    }
```

# ─────────────────────────────────────────────

# DEMO

# ─────────────────────────────────────────────

def demo():
print(”=” * 60)
print(“Claude-Inspired Architecture Demo”)
print(”=” * 60)

```
# Use a small config for demo (full scale needs serious hardware)
config = ClaudeConfig(
    vocab_size=32000,
    hidden_dim=512,       # Small for demo
    num_layers=4,
    num_heads=8,
    num_kv_heads=2,       # GQA: 4 groups
    head_dim=64,
    ffn_dim=1376,
    max_seq_len=2048,
    use_constitutional_filter=True,
)

print(f"\nConfig:")
print(f"  Hidden dim:   {config.hidden_dim}")
print(f"  Layers:       {config.num_layers}")
print(f"  Attn heads:   {config.num_heads}")
print(f"  KV heads:     {config.num_kv_heads} (GQA)")
print(f"  FFN dim:      {config.ffn_dim}")
print(f"  Max seq len:  {config.max_seq_len:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Build model
print("\nBuilding model...")
model = ClaudeModel(config).to(device)

# Test forward pass
print("\nRunning forward pass...")
batch_size = 2
seq_len = 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

with torch.no_grad():
    output = model(input_ids, return_constitutional_scores=True)

print(f"  Input shape:   {input_ids.shape}")
print(f"  Logits shape:  {output['logits'].shape}")
print(f"  Hidden shape:  {output['hidden_states'].shape}")

scores = output["constitutional_scores"]
print(f"\nConstitutional AI Scores (sample 0):")
print(f"  Helpfulness:  {scores['helpfulness'][0].item():.4f}")
print(f"  Harmlessness: {scores['harmlessness'][0].item():.4f}")
print(f"  Honesty:      {scores['honesty'][0].item():.4f}")

# Test generation with KV cache
print("\nTesting autoregressive generation with KV-cache...")
prompt = torch.randint(0, config.vocab_size, (1, 16)).to(device)

generated = model.generate(
    prompt,
    max_new_tokens=32,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
)
print(f"  Prompt tokens:    {prompt.shape[1]}")
print(f"  Generated tokens: {generated.shape[1]}")
print(f"  New tokens:       {generated.shape[1] - prompt.shape[1]}")

# Memory breakdown
total_params = model.count_parameters()
embed_params = sum(p.numel() for p in model.embed_tokens.parameters())
attn_params = sum(
    p.numel() for layer in model.layers
    for p in layer.attn.parameters()
)
ffn_params = sum(
    p.numel() for layer in model.layers
    for p in layer.ffn.parameters()
)

print(f"\nParameter Breakdown:")
print(f"  Total:      {total_params:>12,}")
print(f"  Embeddings: {embed_params:>12,} ({100*embed_params/total_params:.1f}%)")
print(f"  Attention:  {attn_params:>12,} ({100*attn_params/total_params:.1f}%)")
print(f"  FFN:        {ffn_params:>12,} ({100*ffn_params/total_params:.1f}%)")

# Scale to full Claude size
print(f"\n{'─'*40}")
print(f"Full-Scale Estimates (32-layer, 4096-dim):")
full_config = ClaudeConfig()
scale = (full_config.hidden_dim / config.hidden_dim) ** 2
estimated_full = int(total_params * scale * (full_config.num_layers / config.num_layers))
print(f"  Estimated params: ~{estimated_full/1e9:.1f}B")
print(f"  (Claude Sonnet is estimated ~70B parameters)")

print(f"\n{'='*60}")
print("Architecture demo complete.")
print("="*60)
```

if **name** == “**main**”:
demo()
