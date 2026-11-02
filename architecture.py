import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_len, base=10_000):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

    def forward(self, q, k):
        T = q.size(-2)
        cos = self.cos[:, :, :T].to(q.dtype)
        sin = self.sin[:, :, :T].to(q.dtype)

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.value = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)
        self.residual_scale = 1 / math.sqrt(2)

    def forward(self, x, rope):
        B, T, C = x.shape

        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        q, k = rope(q, k)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).reshape(B, T, C)
        x = x + self.out_proj(h) * self.residual_scale

        h = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.value(h)) * self.residual_scale
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ffn_dim, context_length):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, context_length)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffn_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.out_proj.weight, std=std)
            nn.init.normal_(block.down.weight, std=std)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        assert idx.size(1) <= self.context_length
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x, self.rope)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss