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


def block_sparse_mask(seq_len, window_size, num_global, num_random):
    # BigBird-style: local window + a few global tokens + a few random ones
    mask = sliding_window_mask(seq_len, window_size)

    # global tokens see everything and are seen by everything
    mask[:num_global, :] = True
    mask[:, :num_global] = True

    # each row also gets a few random extra positions it can see
    for i in range(seq_len):
        random_cols = torch.randint(0, seq_len, (num_random,))
        mask[i, random_cols] = True

    # still causal, no peeking at future tokens
    return mask & causal_mask(seq_len)
