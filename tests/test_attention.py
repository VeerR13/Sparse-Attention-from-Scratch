import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so we can import src

import torch
import torch.nn.functional as F
from src.attention import dense_attention
from src.masks import causal_mask, sliding_window_mask, block_sparse_mask


def assert_matches_dense_where_masks_agree(dense_mask, sparse_mask, q, k, v):
    dense_out = dense_attention(q, k, v, dense_mask)
    sparse_out = dense_attention(q, k, v, sparse_mask)

    # a row only has to match dense if sparsity didn't actually remove anything from it
    checked = 0
    for i in range(dense_mask.shape[0]):
        if torch.equal(dense_mask[i], sparse_mask[i]):
            assert torch.allclose(dense_out[..., i, :], sparse_out[..., i, :], atol=1e-6)
            checked += 1
    assert checked > 0  # otherwise this test never actually checked anything


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


def test_sliding_window_matches_dense_where_masks_agree():
    torch.manual_seed(0)
    seq_len = 16
    window_size = 4

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)

    dense_mask = causal_mask(seq_len)
    sparse_mask = sliding_window_mask(seq_len, window_size)

    assert_matches_dense_where_masks_agree(dense_mask, sparse_mask, q, k, v)


def test_block_sparse_matches_dense_where_masks_agree():
    torch.manual_seed(0)
    seq_len = 32  # needs enough blocks (8, at block_size=4) or global swallows everything
    block_size = 4

    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)

    dense_mask = causal_mask(seq_len)
    sparse_mask = block_sparse_mask(seq_len, block_size, num_global_blocks=1, num_random_blocks=1)

    assert_matches_dense_where_masks_agree(dense_mask, sparse_mask, q, k, v)
