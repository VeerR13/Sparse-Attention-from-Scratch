import torch
import torch.nn as nn
from src.attention import dense_attention


# one transformer block: attention, residual, layernorm, feedforward, residual, layernorm
class Block(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    # single head on purpose, so we just add and drop a size-1 head dim around dense_attention
    def forward(self, x, mask):
        q = self.q_proj(x).unsqueeze(1)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)

        attn_out = dense_attention(q, k, v, mask).squeeze(1)
        x = self.norm1(x + self.out_proj(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


# 2-layer character-level GPT: token + position embedding, 2 blocks, linear to vocab
class CharGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=128, d_ff=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.block1 = Block(d_model, d_ff)
        self.block2 = Block(d_model, d_ff)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, mask):
        seq_len = idx.shape[1]
        assert seq_len <= self.pos_embed.num_embeddings
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(positions)

        x = self.block1(x, mask)
        x = self.block2(x, mask)

        return self.head(x)
