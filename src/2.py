“””
Claude-Inspired Architecture - Part 2
Extends Part 1 with:

- BPE Tokenizer (like Claude’s tokenizer)
- Constitutional AI training pipeline (RLAIF)
- Speculative decoding for fast inference
- Mixture of Experts (MoE) variant
- Multi-turn conversation handling
- Context window management
- Quantization (INT8/INT4)
- Model sharding for multi-GPU
  “””

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import re
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import struct
import hashlib

# ─────────────────────────────────────────────

# TOKENIZER (BPE - Byte Pair Encoding)

# ─────────────────────────────────────────────

class BPETokenizer:
“””
Byte Pair Encoding tokenizer - same approach Claude uses.
Builds a subword vocabulary by iteratively merging frequent pairs.
“””

```
SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|pad|>": 2,
    "<|unk|>": 3,
    "<|human|>": 4,
    "<|assistant|>": 5,
    "<|system|>": 6,
    "<|tool_result|>": 7,
    "<|tool_call|>": 8,
}

def __init__(self, vocab_size: int = 32000):
    self.vocab_size = vocab_size
    self.vocab: Dict[str, int] = {}
    self.inverse_vocab: Dict[int, str] = {}
    self.merges: Dict[Tuple[str, str], str] = {}
    self.byte_encoder: Dict[int, str] = self._build_byte_encoder()
    self.byte_decoder: Dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

    # Initialize with special tokens
    for token, idx in self.SPECIAL_TOKENS.items():
        self.vocab[token] = idx
        self.inverse_vocab[idx] = token

def _build_byte_encoder(self) -> Dict[int, str]:
    """
    Map bytes to unicode characters.
    This handles all bytes cleanly without needing <unk>.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def _get_pairs(self, word: Tuple[str, ...]) -> set:
    """Get all adjacent pairs in a word"""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs

def train(self, texts: List[str], verbose: bool = True) -> None:
    """
    Train BPE on a corpus of texts.
    Iteratively merge the most frequent adjacent pairs.
    """
    if verbose:
        print(f"Training BPE tokenizer (vocab size: {self.vocab_size})")

    # Build initial character vocabulary from bytes
    word_freqs: Dict[str, int] = defaultdict(int)

    for text in texts:
        # Encode text to bytes, then to unicode chars
        words = re.findall(r'\w+|\s+|[^\w\s]', text)
        for word in words:
            encoded = ''.join(
                self.byte_encoder[b] for b in word.encode('utf-8')
            )
            word_freqs[encoded] += 1

    # Split into characters
    vocab: Dict[Tuple, int] = {}
    for word, freq in word_freqs.items():
        vocab[tuple(word)] = freq

    # Initial char vocab
    char_vocab = set()
    for word in vocab:
        char_vocab.update(word)

    current_vocab_size = len(self.SPECIAL_TOKENS) + len(char_vocab)

    # Add chars to vocab
    for i, char in enumerate(sorted(char_vocab)):
        idx = len(self.SPECIAL_TOKENS) + i
        self.vocab[char] = idx
        self.inverse_vocab[idx] = char

    # BPE merges
    num_merges = self.vocab_size - current_vocab_size
    merge_count = 0

    while merge_count < num_merges:
        # Count pair frequencies
        pair_freqs: Dict[Tuple, int] = defaultdict(int)
        for word, freq in vocab.items():
            pairs = self._get_pairs(word)
            for pair in pairs:
                pair_freqs[pair] += freq

        if not pair_freqs:
            break

        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]

        if best_freq < 2:
            break

        # Create new token
        new_token = best_pair[0] + best_pair[1]
        new_idx = len(self.vocab)
        self.vocab[new_token] = new_idx
        self.inverse_vocab[new_idx] = new_token
        self.merges[best_pair] = new_token

        # Update vocabulary
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and
                    word[i] == best_pair[0] and
                    word[i + 1] == best_pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        vocab = new_vocab

        merge_count += 1
        if verbose and merge_count % 1000 == 0:
            print(f"  Merges: {merge_count}/{num_merges}, vocab size: {len(self.vocab)}")

    if verbose:
        print(f"Training complete. Final vocab size: {len(self.vocab)}")

def _bpe_encode(self, token: str) -> List[str]:
    """Apply BPE merges to a single token"""
    word = tuple(token)
    pairs = self._get_pairs(word)

    if not pairs:
        return list(word)

    while True:
        # Find the highest priority merge
        bigram = min(
            pairs,
            key=lambda pair: self.merges.get(pair, float('inf'))
                if isinstance(self.merges.get(pair, float('inf')), (int, float))
                else float('inf')
        )

        if bigram not in self.merges:
            break

        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(self.merges[bigram])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        pairs = self._get_pairs(word)

        if not pairs:
            break

    return list(word)

def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
    """Encode text to token ids"""
    ids = []

    if add_special_tokens:
        ids.append(self.SPECIAL_TOKENS["<|begin_of_text|>"])

    # Encode byte by byte
    words = re.findall(r'\w+|\s+|[^\w\s]', text)
    for word in words:
        encoded = ''.join(
            self.byte_encoder[b] for b in word.encode('utf-8')
        )
        bpe_tokens = self._bpe_encode(encoded)
        for token in bpe_tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.SPECIAL_TOKENS["<|unk|>"])

    if add_special_tokens:
        ids.append(self.SPECIAL_TOKENS["<|end_of_text|>"])

    return ids

def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
    """Decode token ids back to text"""
    special_ids = set(self.SPECIAL_TOKENS.values())
    tokens = []

    for idx in ids:
        if skip_special_tokens and idx in special_ids:
            continue
        if idx in self.inverse_vocab:
            tokens.append(self.inverse_vocab[idx])

    text = ''.join(tokens)

    # Decode bytes back to utf-8
    byte_array = bytearray([self.byte_decoder.get(c, ord(c)) for c in text])
    return byte_array.decode('utf-8', errors='replace')

def save(self, path: str):
    """Save tokenizer to file"""
    data = {
        "vocab": self.vocab,
        "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
        "vocab_size": self.vocab_size,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load(self, path: str):
    """Load tokenizer from file"""
    with open(path) as f:
        data = json.load(f)
    self.vocab = {k: int(v) for k, v in data["vocab"].items()}
    self.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
    self.merges = {
        tuple(k.split(" ")): v
        for k, v in data["merges"].items()
    }
```

# ─────────────────────────────────────────────

# CONVERSATION / PROMPT FORMATTING

# ─────────────────────────────────────────────

@dataclass
class Message:
role: str   # “system”, “human”, “assistant”
content: str

class ConversationFormatter:
“””
Formats multi-turn conversations into model input.
Claude uses a specific prompt format with role markers.
“””

```
HUMAN_PREFIX = "\n\nHuman: "
ASSISTANT_PREFIX = "\n\nAssistant:"
SYSTEM_PREFIX = "<|system|>"

def format_messages(self, messages: List[Message]) -> str:
    """
    Format conversation history into a prompt string.
    Mirrors Claude's actual prompt format.
    """
    prompt = ""

    for msg in messages:
        if msg.role == "system":
            prompt += f"{self.SYSTEM_PREFIX}\n{msg.content}\n"
        elif msg.role == "human":
            prompt += f"{self.HUMAN_PREFIX}{msg.content}"
        elif msg.role == "assistant":
            prompt += f"{self.ASSISTANT_PREFIX} {msg.content}"

    # Add assistant turn start for generation
    prompt += self.ASSISTANT_PREFIX

    return prompt

def extract_response(self, full_text: str) -> str:
    """Extract just the last assistant response"""
    parts = full_text.split(self.ASSISTANT_PREFIX)
    if len(parts) > 1:
        response = parts[-1].strip()
        # Stop at next Human turn
        if self.HUMAN_PREFIX in response:
            response = response.split(self.HUMAN_PREFIX)[0].strip()
        return response
    return full_text
```

# ─────────────────────────────────────────────

# CONSTITUTIONAL AI (RLAIF)

# ─────────────────────────────────────────────

class ConstitutionalPrinciples:
“””
The principles Claude is trained on.
In real Claude, these guide the RLAIF (RL from AI Feedback) process.
“””

```
PRINCIPLES = [
    # Helpfulness
    "Choose the response that is most helpful to the human.",
    "Choose the response that best follows the human's instructions.",
    "Choose the response that provides the most useful information.",

    # Harmlessness
    "Choose the response that is least likely to cause harm.",
    "Choose the response that avoids dangerous or illegal content.",
    "Choose the response that respects human rights and dignity.",
    "Choose the response that does not assist with weapons or violence.",

    # Honesty
    "Choose the response that is most honest and truthful.",
    "Choose the response that avoids deception.",
    "Choose the response that acknowledges uncertainty when appropriate.",
    "Choose the response that does not impersonate real people.",

    # Ethics
    "Choose the response that best embodies ethical behavior.",
    "Choose the response that treats all people fairly.",
    "Choose the response that supports human autonomy.",
]

CRITIQUE_REQUEST = (
    "Identify specific ways in which the assistant's last response "
    "is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
)

REVISION_REQUEST = (
    "Please rewrite the assistant response to remove any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content."
)
```

class RLAIFTrainer:
“””
Reinforcement Learning from AI Feedback.
Claude uses an AI model to generate preference labels instead of
(or in addition to) human labelers.

```
Pipeline:
1. Generate multiple responses to a prompt
2. Use a "preference model" (another Claude) to rank them
3. Use rankings to train reward model
4. Use reward model in PPO
"""

def __init__(self, principles: ConstitutionalPrinciples):
    self.principles = principles

def critique_and_revise(
    self,
    original_prompt: str,
    original_response: str,
    policy_model,  # The model being trained
    tokenizer: BPETokenizer,
    num_revisions: int = 2,
) -> List[str]:
    """
    Constitutional AI critique-revision cycle.
    Returns list of progressively improved responses.
    """
    responses = [original_response]
    current = original_response

    for i in range(num_revisions):
        # Build critique prompt
        critique_prompt = (
            f"{original_prompt}\n\n"
            f"Response: {current}\n\n"
            f"Critique: {self.principles.CRITIQUE_REQUEST}\n\n"
            f"Critique:"
        )

        # Build revision prompt
        revision_prompt = (
            f"{original_prompt}\n\n"
            f"Original response: {original_response}\n\n"
            f"Revision request: {self.principles.REVISION_REQUEST}\n\n"
            f"Revised response:"
        )

        # In practice: run through policy model to generate critique + revision
        # Here we return the prompts for use with actual generation
        responses.append(revision_prompt)

    return responses

def rank_responses(
    self,
    prompt: str,
    responses: List[str],
    principle: str,
) -> List[int]:
    """
    Rank responses by a constitutional principle.
    Returns indices sorted best to worst.

    In real RLAIF: a preference model does this ranking.
    """
    # Placeholder - real impl uses a trained preference model
    # Returns random ranking for demo purposes
    indices = list(range(len(responses)))
    np.random.shuffle(indices)
    return indices

def generate_preference_pairs(
    self,
    prompt: str,
    responses: List[str],
) -> List[Tuple[str, str]]:
    """
    Generate (chosen, rejected) pairs for reward model training.
    """
    pairs = []

    for principle in self.principles.PRINCIPLES:
        rankings = self.rank_responses(prompt, responses, principle)
        if len(rankings) >= 2:
            chosen = responses[rankings[0]]
            rejected = responses[rankings[-1]]
            pairs.append((chosen, rejected))

    return pairs
```

# ─────────────────────────────────────────────

# SPECULATIVE DECODING

# ─────────────────────────────────────────────

class SpeculativeDecoder:
“””
Speculative decoding for faster inference.

```
How it works:
1. Small "draft" model generates K tokens quickly
2. Large "target" model (Claude) verifies them in parallel
3. Accept tokens where both models agree, reject where they differ
4. Net result: ~3x speedup with identical output distribution

This is likely used in Claude's inference pipeline.
"""

def __init__(
    self,
    target_model,    # Large model (Claude)
    draft_model,     # Small fast model
    num_draft_tokens: int = 4,
    acceptance_threshold: float = 0.7,
):
    self.target = target_model
    self.draft = draft_model
    self.K = num_draft_tokens
    self.threshold = acceptance_threshold

@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Generate tokens using speculative decoding.
    """
    generated = input_ids.clone()
    accepted_total = 0
    draft_total = 0

    while generated.shape[1] < input_ids.shape[1] + max_new_tokens:

        # Step 1: Draft model generates K tokens
        draft_tokens = []
        draft_probs = []
        draft_input = generated.clone()

        for _ in range(self.K):
            draft_out = self.draft(draft_input)
            draft_logits = draft_out["logits"][:, -1, :] / max(temperature, 1e-8)
            draft_p = F.softmax(draft_logits, dim=-1)
            draft_token = torch.multinomial(draft_p, 1)
            draft_tokens.append(draft_token)
            draft_probs.append(draft_p)
            draft_input = torch.cat([draft_input, draft_token], dim=1)

        draft_sequence = torch.cat(draft_tokens, dim=1)

        # Step 2: Target model verifies all K tokens in parallel
        verify_input = torch.cat([generated, draft_sequence], dim=1)
        target_out = self.target(verify_input)
        target_logits = target_out["logits"][:, generated.shape[1] - 1:-1, :]

        # Step 3: Acceptance/rejection sampling
        accepted = 0
        for k in range(self.K):
            target_p = F.softmax(target_logits[:, k, :] / max(temperature, 1e-8), dim=-1)
            draft_p = draft_probs[k]
            token = draft_tokens[k]

            # Acceptance probability
            accept_prob = torch.min(
                torch.ones_like(draft_p),
                target_p / (draft_p + 1e-10)
            ).gather(1, token)

            if torch.rand(1).item() < accept_prob.item():
                generated = torch.cat([generated, token], dim=1)
                accepted += 1
            else:
                # Reject: sample from corrected distribution
                corrected = torch.clamp(target_p - draft_p, min=0)
                corrected = corrected / corrected.sum()
                bonus_token = torch.multinomial(corrected, 1)
                generated = torch.cat([generated, bonus_token], dim=1)
                break

        accepted_total += accepted
        draft_total += self.K

    acceptance_rate = accepted_total / max(draft_total, 1)
    return generated, acceptance_rate
```

# ─────────────────────────────────────────────

# MIXTURE OF EXPERTS (MoE) VARIANT

# ─────────────────────────────────────────────

class ExpertFFN(nn.Module):
“”“Single expert - same as SwiGLU FFN”””

```
def __init__(self, hidden_dim: int, ffn_dim: int):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
    self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
    self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

class MixtureOfExperts(nn.Module):
“””
Sparse Mixture of Experts FFN layer.
Each token is routed to top-K experts.

```
This is the architecture used in models like Mixtral.
Claude may use something similar in larger versions.
"""

def __init__(
    self,
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int = 8,
    top_k: int = 2,
):
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k
    self.hidden_dim = hidden_dim

    # Router: decides which experts to use
    self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    # Expert pool
    self.experts = nn.ModuleList([
        ExpertFFN(hidden_dim, ffn_dim)
        for _ in range(num_experts)
    ])

def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    x_flat = x.view(-1, D)  # (B*T, D)

    # Compute routing weights
    router_logits = self.router(x_flat)  # (B*T, num_experts)
    routing_weights = F.softmax(router_logits, dim=-1)

    # Select top-k experts
    top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    # Compute expert outputs
    output = torch.zeros_like(x_flat)
    load_balance_loss = torch.tensor(0.0, device=x.device)

    for expert_idx in range(self.num_experts):
        # Find tokens routed to this expert
        expert_mask = (top_k_indices == expert_idx).any(dim=-1)
        if not expert_mask.any():
            continue

        expert_input = x_flat[expert_mask]
        expert_output = self.experts[expert_idx](expert_input)

        # Weight by routing probability
        expert_weights = top_k_weights[expert_mask]
        for k in range(self.top_k):
            k_mask = top_k_indices[expert_mask, k] == expert_idx
            if k_mask.any():
                weight = expert_weights[k_mask, k].unsqueeze(-1)
                output[expert_mask] = output[expert_mask].clone()
                output[expert_mask][k_mask] += weight * expert_output[k_mask]

    # Load balancing loss (prevents all tokens routing to same expert)
    avg_routing = routing_weights.mean(dim=0)
    ideal = torch.ones_like(avg_routing) / self.num_experts
    load_balance_loss = F.mse_loss(avg_routing, ideal)

    return output.view(B, T, D), load_balance_loss
```

# ─────────────────────────────────────────────

# INT8 QUANTIZATION

# ─────────────────────────────────────────────

class QuantizedLinear(nn.Module):
“””
INT8 quantized linear layer.
Claude’s inference likely uses quantization for efficiency.
~4x memory reduction, ~2x speed on compatible hardware.
“””

```
def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    # Store weights as INT8
    self.register_buffer(
        'weight_int8',
        torch.zeros(out_features, in_features, dtype=torch.int8)
    )
    self.register_buffer(
        'weight_scale',
        torch.ones(out_features, dtype=torch.float32)
    )

@classmethod
def from_linear(cls, linear: nn.Linear) -> 'QuantizedLinear':
    """Convert a standard Linear layer to quantized INT8"""
    q = cls(linear.in_features, linear.out_features)

    # Per-channel quantization
    weight = linear.weight.detach().float()
    scales = weight.abs().max(dim=1)[0] / 127.0
    scales = scales.clamp(min=1e-8)

    weight_int8 = (weight / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    q.weight_int8 = weight_int8
    q.weight_scale = scales

    return q

def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Dequantize weights
    weight = self.weight_int8.float() * self.weight_scale.unsqueeze(1)
    return F.linear(x, weight)
```

def quantize_model(model: nn.Module, skip_layers: List[str] = None) -> nn.Module:
“””
Quantize all Linear layers in a model to INT8.
Skips embedding and LM head layers by default.
“””
skip_layers = skip_layers or [‘embed_tokens’, ‘lm_head’]

```
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # Check if should skip
        should_skip = any(skip in name for skip in skip_layers)
        if should_skip:
            continue

        # Replace with quantized version
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model if not parent_name else dict(model.named_modules())[parent_name]
        quantized = QuantizedLinear.from_linear(module)
        setattr(parent, child_name, quantized)

return model
```

# ─────────────────────────────────────────────

# CONTEXT WINDOW MANAGEMENT

# ─────────────────────────────────────────────

class ContextWindowManager:
“””
Manages Claude’s very long context windows (up to 200k tokens).

```
Strategies:
- Sliding window: Keep most recent tokens
- Summarization: Compress old context
- Retrieval: Index old context, retrieve relevant parts
"""

def __init__(
    self,
    max_tokens: int = 200000,
    summarize_threshold: float = 0.8,
):
    self.max_tokens = max_tokens
    self.summarize_threshold = summarize_threshold
    self.context_chunks: List[Dict] = []
    self.total_tokens: int = 0

def add_message(self, message: Message, token_count: int):
    """Add a message to context, managing overflow"""
    self.context_chunks.append({
        "message": message,
        "tokens": token_count,
        "position": self.total_tokens,
    })
    self.total_tokens += token_count

    # Check if we need to manage context
    if self.total_tokens > self.max_tokens * self.summarize_threshold:
        self._compress_context()

def _compress_context(self):
    """
    Compress old context when approaching limit.
    In real Claude: uses a separate summarization model/call.
    """
    # Keep system prompt + recent messages
    keep_recent = max(4, len(self.context_chunks) // 2)
    old_chunks = self.context_chunks[:-keep_recent]
    self.context_chunks = self.context_chunks[-keep_recent:]

    # Summarize old content
    old_text = " ".join(
        chunk["message"].content for chunk in old_chunks
    )
    summary_text = f"[Earlier conversation summary: {old_text[:200]}...]"

    summary_message = Message(role="system", content=summary_text)
    summary_tokens = len(summary_text.split()) * 2  # rough estimate

    # Insert summary at beginning
    self.context_chunks.insert(0, {
        "message": summary_message,
        "tokens": summary_tokens,
        "position": 0,
    })

    # Recompute total
    self.total_tokens = sum(c["tokens"] for c in self.context_chunks)

def get_messages(self) -> List[Message]:
    """Get current context as list of messages"""
    return [chunk["message"] for chunk in self.context_chunks]

@property
def utilization(self) -> float:
    return self.total_tokens / self.max_tokens
```

# ─────────────────────────────────────────────

# COMPLETE INFERENCE ENGINE

# ─────────────────────────────────────────────

class ClaudeInferenceEngine:
“””
Complete inference engine combining all components.
This is the full pipeline from text input to text output.
“””

```
def __init__(
    self,
    model,
    tokenizer: BPETokenizer,
    formatter: ConversationFormatter,
    max_context: int = 200000,
    use_speculative: bool = False,
    draft_model=None,
):
    self.model = model
    self.tokenizer = tokenizer
    self.formatter = formatter
    self.context_manager = ContextWindowManager(max_context)
    self.use_speculative = use_speculative
    self.draft_model = draft_model

    if use_speculative and draft_model:
        self.speculative_decoder = SpeculativeDecoder(model, draft_model)

    self.conversation_history: List[Message] = []

def add_system_prompt(self, system_prompt: str):
    """Set the system prompt"""
    msg = Message(role="system", content=system_prompt)
    token_count = len(self.tokenizer.encode(system_prompt))
    self.conversation_history.insert(0, msg)
    self.context_manager.add_message(msg, token_count)

@torch.no_grad()
def chat(
    self,
    user_message: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    stream: bool = False,
) -> str:
    """
    Process a user message and generate a response.
    """
    # Add user message
    user_msg = Message(role="human", content=user_message)
    self.conversation_history.append(user_msg)
    self.context_manager.add_message(
        user_msg, len(self.tokenizer.encode(user_message))
    )

    # Format conversation
    messages = self.context_manager.get_messages()
    prompt = self.formatter.format_messages(messages)

    # Tokenize
    input_ids = torch.tensor(
        [self.tokenizer.encode(prompt)],
        dtype=torch.long
    )

    # Generate
    if self.use_speculative and self.draft_model:
        output_ids, acceptance_rate = self.speculative_decoder.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature
        )
    else:
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # Decode
    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = self.tokenizer.decode(new_tokens.tolist())
    response = self.formatter.extract_response(response)

    # Add to history
    assistant_msg = Message(role="assistant", content=response)
    self.conversation_history.append(assistant_msg)
    self.context_manager.add_message(
        assistant_msg, len(new_tokens)
    )

    return response

def reset_conversation(self):
    """Clear conversation history"""
    self.conversation_history = []
    self.context_manager = ContextWindowManager(
        self.context_manager.max_tokens
    )

def get_context_stats(self) -> dict:
    """Return context window usage stats"""
    return {
        "total_tokens": self.context_manager.total_tokens,
        "max_tokens": self.context_manager.max_tokens,
        "utilization": f"{self.context_manager.utilization:.1%}",
        "num_messages": len(self.conversation_history),
    }
```

# ─────────────────────────────────────────────

# MODEL SHARDING (Multi-GPU)

# ─────────────────────────────────────────────

class ModelSharder:
“””
Shard a large model across multiple GPUs.
Claude at full scale requires multiple GPUs / TPUs.

```
Pipeline parallelism: Different layers on different devices.
Tensor parallelism: Split individual weight matrices.
"""

@staticmethod
def pipeline_parallel(model: nn.Module, num_gpus: int) -> nn.Module:
    """
    Distribute transformer layers across GPUs.
    GPU 0: embedding + layers 0..N/K
    GPU 1: layers N/K..2N/K
    ...
    GPU K-1: last layers + head
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        print(f"Warning: {num_gpus} GPUs requested but not available. Using CPU.")
        return model

    num_layers = len(model.layers)
    layers_per_gpu = math.ceil(num_layers / num_gpus)

    # Move embeddings to GPU 0
    model.embed_tokens = model.embed_tokens.to(f'cuda:0')

    # Distribute layers
    for i, layer in enumerate(model.layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        model.layers[i] = layer.to(f'cuda:{gpu_id}')

    # Move head to last GPU
    last_gpu = num_gpus - 1
    model.norm = model.norm.to(f'cuda:{last_gpu}')
    model.lm_head = model.lm_head.to(f'cuda:{last_gpu}')

    print(f"Model distributed across {num_gpus} GPUs")
    print(f"  {layers_per_gpu} layers per GPU")

    return model

@staticmethod
def get_memory_estimate(config) -> dict:
    """
    Estimate memory requirements for the model.
    """
    # Parameters: 2 bytes (fp16) or 4 bytes (fp32) per param
    embed_params = config.vocab_size * config.hidden_dim
    attn_params_per_layer = (
        config.hidden_dim * config.num_heads * config.head_dim +  # Q
        config.hidden_dim * config.num_kv_heads * config.head_dim * 2 +  # K, V
        config.num_heads * config.head_dim * config.hidden_dim  # O
    )
    ffn_params_per_layer = (
        config.hidden_dim * config.ffn_dim * 3  # gate, up, down
    )
    total_params = (
        embed_params +
        config.num_layers * (attn_params_per_layer + ffn_params_per_layer)
    )

    return {
        "total_params": f"{total_params/1e9:.2f}B",
        "fp32_memory_gb": f"{total_params * 4 / 1e9:.1f} GB",
        "fp16_memory_gb": f"{total_params * 2 / 1e9:.1f} GB",
        "int8_memory_gb": f"{total_params * 1 / 1e9:.1f} GB",
        "kv_cache_per_token_mb": f"{config.num_kv_heads * config.head_dim * config.num_layers * 2 * 2 / 1e6:.2f} MB",
    }
```

# ─────────────────────────────────────────────

# DEMO

# ─────────────────────────────────────────────

def demo_tokenizer():
print(”\n” + “=” * 60)
print(“BPE Tokenizer Demo”)
print(”=” * 60)

```
tokenizer = BPETokenizer(vocab_size=500)

# Train on small corpus
corpus = [
    "Hello world! This is a test of the tokenizer.",
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks learn representations from data.",
    "Transformers use attention mechanisms for sequence modeling.",
    "Claude is a large language model built by Anthropic.",
    "Constitutional AI helps ensure safe and helpful behavior.",
    "Brain computer interfaces allow direct neural communication.",
    "Machine learning models are trained on large datasets.",
] * 10  # Repeat for more training signal

tokenizer.train(corpus, verbose=False)

# Test encoding
test_text = "Hello, Claude!"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"Input:   '{test_text}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{decoded}'")
print(f"Vocab size: {len(tokenizer.vocab)}")
```

def demo_conversation():
print(”\n” + “=” * 60)
print(“Conversation Formatter Demo”)
print(”=” * 60)

```
formatter = ConversationFormatter()

messages = [
    Message(role="system", content="You are Claude, a helpful AI assistant built by Anthropic."),
    Message(role="human", content="What is constitutional AI?"),
    Message(role="assistant", content="Constitutional AI is a training approach developed by Anthropic..."),
    Message(role="human", content="How does it work?"),
]

prompt = formatter.format_messages(messages)
print(prompt[:500] + "...")
```

def demo_moe():
print(”\n” + “=” * 60)
print(“Mixture of Experts Demo”)
print(”=” * 60)

```
moe = MixtureOfExperts(
    hidden_dim=512,
    ffn_dim=1376,
    num_experts=8,
    top_k=2,
)

x = torch.randn(2, 32, 512)
output, lb_loss = moe(x)

print(f"Input shape:      {x.shape}")
print(f"Output shape:     {output.shape}")
print(f"Load balance loss: {lb_loss.item():.4f}")
print(f"Expert params:    {sum(p.numel() for p in moe.parameters()):,}")
print(f"Active params:    ~{sum(p.numel() for p in moe.parameters()) * 2 // 8:,} (top-2 of 8)")
```

def demo_context_manager():
print(”\n” + “=” * 60)
print(“Context Window Manager Demo”)
print(”=” * 60)

```
ctx = ContextWindowManager(max_tokens=1000)

# Simulate conversation
for i in range(20):
    msg = Message(
        role="human" if i % 2 == 0 else "assistant",
        content=f"Message {i}: " + "token " * 30
    )
    ctx.add_message(msg, 35)

print(f"Messages added:  20")
print(f"Messages kept:   {len(ctx.context_chunks)}")
print(f"Total tokens:    {ctx.total_tokens}")
print(f"Utilization:     {ctx.utilization:.1%}")
```

def demo_memory_estimates():
print(”\n” + “=” * 60)
print(“Memory Estimates for Different Model Sizes”)
print(”=” * 60)

```
configs = [
    ("Claude Haiku (est.)", {"vocab_size": 32000, "hidden_dim": 2048, "num_layers": 24,
                               "num_heads": 16, "num_kv_heads": 4, "head_dim": 128, "ffn_dim": 5504}),
    ("Claude Sonnet (est.)", {"vocab_size": 32000, "hidden_dim": 4096, "num_layers": 32,
                                "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 11008}),
    ("Claude Opus (est.)", {"vocab_size": 32000, "hidden_dim": 8192, "num_layers": 80,
                              "num_heads": 64, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 28672}),
]

for name, cfg in configs:
    class C:
        pass
    c = C()
    for k, v in cfg.items():
        setattr(c, k, v)

    mem = ModelSharder.get_memory_estimate(c)
    print(f"\n{name}:")
    for k, v in mem.items():
        print(f"  {k}: {v}")
```

def run_all_demos():
print(”=” * 60)
print(“Claude Architecture v2 - Extended Components”)
print(”=” * 60)

```
demo_tokenizer()
demo_conversation()
demo_moe()
demo_context_manager()
demo_memory_estimates()

print("\n" + "=" * 60)
print("All demos complete.")
print("=" * 60)
```

if **name** == “**main**”:
run_all_demos()
