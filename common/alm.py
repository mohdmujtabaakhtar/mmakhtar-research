"""Audio-conditioned language-model head (SATYAM, GARUDA).

Audio features are projected into a causal language model's embedding space and
read as a *continuous prefix*, followed by a text prompt. The LM then answers in
one word, and training uses the ordinary next-token loss on that answer:

    [ prefix_1 ... prefix_P | "Is the speech sample fake or real? ..." | answer ]

Decoding is restricted to the candidate answers (e.g. "Real" / "Fake"): the
class score is the answer's log-likelihood, so ``softmax(logits)`` gives the
probability of each answer and ``logits[:, fake] - logits[:, real]`` is the
detection score used for EER.

Backbones:

* any Hugging Face causal LM, e.g. ``Qwen/Qwen2-7B`` or ``Qwen/Qwen2-0.5B``, kept
  frozen, optionally with LoRA adapters on the attention projections;
* ``toy``: a tiny randomly initialised causal transformer with a word-level
  tokenizer, used by the smoke tests so they run offline on a CPU.
"""
from __future__ import annotations

import math
import re
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and A, B trainable (B starts at zero)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype,
                                          device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                          device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.dropout(x) @ self.A.T @ self.B.T * self.scale


def apply_lora(model: nn.Module, targets=("q_proj", "v_proj"), rank: int = 8, alpha: float = 32.0) -> int:
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets``; returns how many were wrapped."""
    count = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                count += 1
    return count


# ----------------------------------------------------------------- toy backbone

class WordTokenizer:
    """Lower-cased word/punctuation tokenizer over a fixed vocabulary."""

    def __init__(self, texts):
        words = sorted({w for t in texts for w in self.split(t)})
        self.vocab = {w: i + 1 for i, w in enumerate(words)}   # 0 = unknown

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(w, 0) for w in self.split(text)]

    def __len__(self) -> int:
        return len(self.vocab) + 1


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q_proj, self.k_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.v_proj, self.o_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = (p(h).view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.norm2(x))


class TinyCausalLM(nn.Module):
    """Small pre-norm causal transformer with a Hugging Face-like interface."""

    def __init__(self, vocab_size: int, hidden: int = 64, layers: int = 2, heads: int = 4,
                 max_len: int = 256):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Parameter(torch.randn(max_len, hidden) * 0.02)
        self.blocks = nn.ModuleList([_Block(hidden, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, inputs_embeds: torch.Tensor, output_hidden_states: bool = False, **_):
        x = inputs_embeds + self.pos[: inputs_embeds.shape[1]]
        hidden = [x]
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)),
                               hidden_states=tuple(hidden) if output_hidden_states else None)


# ----------------------------------------------------------------- prefix head

class PrefixLMHead(nn.Module):
    """Frozen causal LM that reads audio prefix tokens + a prompt and scores candidate answers."""

    def __init__(self, backbone: str = "toy", prompt: str = "Is the speech real or fake?",
                 answers=("Real", "Fake"), lora_rank: int = 0, lora_alpha: float = 32.0,
                 lora_targets=("q_proj", "v_proj"), dtype: str = "float32", toy_hidden: int = 64,
                 toy_layers: int = 2, extra_texts=(), tokenizer: str | None = None):
        super().__init__()
        self.backbone, self.answers = backbone, list(answers)
        if backbone == "toy":
            self.tokenizer = WordTokenizer([prompt, *answers, *extra_texts])
            generator_state = torch.random.get_rng_state()
            torch.manual_seed(1234)          # the "pretrained" toy LM is the same in every run
            self.lm = TinyCausalLM(len(self.tokenizer), toy_hidden, toy_layers)
            torch.random.set_rng_state(generator_state)
            encode = self.tokenizer.encode
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or backbone)
            self.lm = AutoModelForCausalLM.from_pretrained(backbone, dtype=getattr(torch, dtype))
            encode = lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
        for p in self.lm.parameters():
            p.requires_grad_(False)
        if lora_rank:
            apply_lora(self.lm, tuple(lora_targets), lora_rank, lora_alpha)
        self.hidden_size = self.lm.config.hidden_size
        self._encode = encode
        ids = lambda text: torch.tensor(encode(text), dtype=torch.long)   # noqa: E731
        self.register_buffer("prompt_ids", ids(prompt), persistent=False)
        self.answer_ids = [ids(" " + a if backbone != "toy" else a) for a in answers]
        if any(len(a) == 0 for a in self.answer_ids) or len(self.prompt_ids) == 0:
            raise ValueError("the tokenizer produced an empty prompt or answer")

    @property
    def dtype(self) -> torch.dtype:
        return self.lm.get_input_embeddings().weight.dtype

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids.to(self.prompt_ids.device))

    @torch.no_grad()
    def text_representation(self, text: str, layer: int = -1) -> torch.Tensor:
        """Mean-pooled hidden states of ``text`` at one layer of the LM, shape (hidden,)."""
        ids = torch.tensor(self._encode(text), dtype=torch.long, device=self.prompt_ids.device)[None]
        out = self.lm(inputs_embeds=self.embed_ids(ids), output_hidden_states=True)
        return out.hidden_states[layer][0].float().mean(0)

    def forward(self, prefix: torch.Tensor, target: torch.Tensor | None = None) -> dict:
        """prefix (B, P, hidden) -> answer log-likelihoods (B, n_answers) and the LM loss."""
        b = prefix.shape[0]
        prompt = self.embed_ids(self.prompt_ids)[None].expand(b, -1, -1)
        start = prefix.shape[1] + prompt.shape[1] - 1      # position that predicts the first answer token
        scores, token_nll = [], []
        for ids in self.answer_ids:
            ids = ids.to(prefix.device)
            answer = self.embed_ids(ids[:-1])[None].expand(b, -1, -1)
            seq = torch.cat([prefix.to(self.dtype), prompt, answer], dim=1)
            logits = self.lm(inputs_embeds=seq).logits[:, start:start + len(ids)].float()
            logp = logits.log_softmax(-1).gather(-1, ids[None, :, None].expand(b, -1, 1)).squeeze(-1)
            scores.append(logp.sum(-1))
            token_nll.append(-logp.mean(-1))
        scores = torch.stack(scores, dim=1)
        out = {"logits": scores}
        if target is not None:
            out["lm_loss"] = torch.stack(token_nll, 1).gather(1, target[:, None]).mean()
        return out
