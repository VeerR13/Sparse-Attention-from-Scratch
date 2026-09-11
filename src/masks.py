import torch


def causal_mask(seq_len):
    # True means allowed, this is a lower triangle so position i only sees 0..i
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def sliding_window_mask(seq_len, window_size):
    # like causal, but position i also forgets anything older than window_size steps back
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i + 1] = True
    return mask


def block_sparse_mask(seq_len, block_size, num_global_blocks, num_random_blocks, generator=None):
    # BigBird-style: decide per BLOCK first, then expand to tokens
    num_blocks = (seq_len + block_size - 1) // block_size  # round up

    block_mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)

    for i in range(num_blocks):
        # local window: this block plus one either side
        for j in range(max(0, i - 1), min(num_blocks, i + 2)):
            block_mask[i, j] = True

        # random: only sample from blocks i is allowed to see, so nothing gets wasted later
        legal = list(range(i + 1))
        k = min(num_random_blocks, len(legal))
        picks = torch.randperm(len(legal), generator=generator)[:k]
        for p in picks:
            block_mask[i, legal[p]] = True

    # global: first and last block see everything and are seen by everything
    # guarded because -0 is 0 in a slice, so block_mask[-0:] would mean "everything"
    if num_global_blocks > 0:
        block_mask[:num_global_blocks, :] = True
        block_mask[:, :num_global_blocks] = True
        block_mask[-num_global_blocks:, :] = True
        block_mask[:, -num_global_blocks:] = True

    mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    mask = mask[:seq_len, :seq_len]  # trim the extra from rounding up to a full block

    # required, not just a check: window and global above are built symmetric on purpose,
    # this is what cuts them down to their causal, one-sided shape
    return mask & causal_mask(seq_len)
