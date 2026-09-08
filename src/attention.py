import torch
import torch.nn.functional as F


def dense_attention(q, k, v, mask=None):
    # q, k, v shape: (batch, heads, seq_len, head_dim)
    head_dim = q.shape[-1]

    scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)

    if mask is not None:
        # mask is boolean, True means "allowed to attend", False means "blocked"
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out
