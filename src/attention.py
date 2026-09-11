import torch
import torch.nn.functional as F


# dense attention: matmul, mask, softmax, matmul
def dense_attention(q, k, v, mask=None):
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)

    if mask is not None:
        # mask is boolean, True means allowed to attend, False means blocked
        empty = ~mask.any(dim=-1, keepdim=True)

        # a token allowed to attend to nothing gets zero output instead of NaN from softmax over all -inf
        scores = scores.masked_fill(~mask & ~empty, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(empty, torch.zeros_like(weights), weights)
    else:
        weights = F.softmax(scores, dim=-1)

    out = weights @ v
    return out
