import torch


# causal mask, position i only sees 0..i
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


# causal mask that also forgets anything older than window_size steps back
def sliding_window_mask(seq_len, window_size):
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i + 1] = True
    return mask


# BigBird-style mask: local window + global blocks + random blocks, decided per block then expanded to tokens
def block_sparse_mask(seq_len, block_size, num_global_blocks, num_random_blocks, generator=None):
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    num_blocks = (seq_len + block_size - 1) // block_size
    block_mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)

    for i in range(num_blocks):
        for j in range(max(0, i - 1), min(num_blocks, i + 2)):
            block_mask[i, j] = True

        legal = list(range(i + 1))
        k = min(num_random_blocks, len(legal))
        picks = torch.randperm(len(legal), generator=generator)[:k]
        for p in picks:
            block_mask[i, legal[p]] = True

    # guard is required: -0 is 0 in a slice, so block_mask[-0:, :] would mean "everything", not "nothing"
    if num_global_blocks > 0:
        block_mask[:num_global_blocks, :] = True
        block_mask[:, :num_global_blocks] = True
        block_mask[-num_global_blocks:, :] = True
        block_mask[:, -num_global_blocks:] = True

    mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    mask = mask[:seq_len, :seq_len]

    # required, not redundant: window and global above are built symmetric, this performs the causal collapse
    return mask & causal_mask(seq_len)
