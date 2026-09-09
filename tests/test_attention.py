import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so we can import src

import torch
import torch.nn.functional as F
from src.attention import dense_attention


def test_dense_attention_matches_reference():
    torch.manual_seed(0)

    batch = 2
    heads = 4
    seq_len = 16
    head_dim = 8

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  # causal: True means allowed

    my_out = dense_attention(q, k, v, mask)

    # this is the one allowed use of the banned function, only here, as a check against ours
    ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    assert torch.allclose(my_out, ref_out, atol=1e-6)
