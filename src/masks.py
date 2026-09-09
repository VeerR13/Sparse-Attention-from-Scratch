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
