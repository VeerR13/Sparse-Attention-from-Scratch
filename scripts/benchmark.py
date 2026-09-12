import os
import csv
import json
import time
import torch
import matplotlib.pyplot as plt
from src.attention import dense_attention
from src.masks import causal_mask, sliding_window_mask, block_sparse_mask

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
WINDOW_SIZE = 256
BLOCK_SIZE = 64
NUM_GLOBAL_BLOCKS = 1
NUM_RANDOM_BLOCKS = 3


# random q, k, v for one head and one batch, so seq_len is the only variable
def make_qkv(seq_len, device):
    torch.manual_seed(0)
    q = torch.randn(1, 1, seq_len, 64, device=device)
    k = torch.randn(1, 1, seq_len, 64, device=device)
    v = torch.randn(1, 1, seq_len, 64, device=device)
    return q, k, v


# median wall-clock time for one forward pass, in milliseconds
def median_time_ms(mask, q, k, v, device, warmup=3, repeats=8):
    with torch.no_grad():
        for _ in range(warmup):
            dense_attention(q, k, v, mask)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            dense_attention(q, k, v, mask)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


# peak memory for one forward pass, in megabytes. only meaningful on cuda
def peak_memory_mb(mask, q, k, v, device):
    if device != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        dense_attention(q, k, v, mask)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1e6


# GPU name, torch version, CUDA version, so the plots are self-describing
def hardware_info():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    else:
        gpu = "CPU only, memory numbers are 0"
        cuda_version = "n/a"
    return {"gpu": gpu, "torch_version": torch.__version__, "cuda_version": cuda_version}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = hardware_info()
    print("hardware:", info)

    patterns = {
        "dense": lambda n: causal_mask(n),
        "sliding_window": lambda n: sliding_window_mask(n, WINDOW_SIZE),
        "block_sparse": lambda n: block_sparse_mask(n, BLOCK_SIZE, NUM_GLOBAL_BLOCKS, NUM_RANDOM_BLOCKS),
    }

    rows = []
    for seq_len in SEQ_LENS:
        q, k, v = make_qkv(seq_len, device)
        for name, mask_fn in patterns.items():
            mask = mask_fn(seq_len).to(device)
            ms = median_time_ms(mask, q, k, v, device)
            mem = peak_memory_mb(mask, q, k, v, device)
            cells = mask.sum().item()
            rows.append({"pattern": name, "seq_len": seq_len, "time_ms": ms, "peak_mem_mb": mem, "cells": cells})
            print(f"{name:15s} seq_len={seq_len:5d}  {ms:8.3f} ms  {mem:9.2f} MB  {cells:>10} cells")

    save_results(rows, info)
    plot_measured(rows, info)
    plot_theoretical(rows)


# raw numbers to results/, so the writeup can quote them without re-running
def save_results(rows, info):
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    with open("results/benchmark.json", "w") as f:
        json.dump({"hardware": info, "rows": rows}, f, indent=2)

    with open("results/benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern", "seq_len", "time_ms", "peak_mem_mb", "cells"])
        writer.writeheader()
        writer.writerows(rows)


# figure A: measured wall-clock and peak memory vs seq_len, per pattern
def plot_measured(rows, info):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in ["dense", "sliding_window", "block_sparse"]:
        xs = [r["seq_len"] for r in rows if r["pattern"] == name]
        ax1.plot(xs, [r["time_ms"] for r in rows if r["pattern"] == name], marker="o", label=name)
        ax2.plot(xs, [r["peak_mem_mb"] for r in rows if r["pattern"] == name], marker="o", label=name)

    for ax, ylabel, title in [(ax1, "time per forward pass (ms)", "measured: wall-clock"),
                               (ax2, "peak memory (MB)", "measured: peak memory")]:
        ax.set_xlabel("sequence length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

    fig.suptitle(f"{info['gpu']} | torch {info['torch_version']} | cuda {info['cuda_version']}")
    fig.tight_layout()
    fig.savefig("results/plots/benchmark_measured.png")


# figure B: cells each mask actually allows, vs seq_len, per pattern
def plot_theoretical(rows):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["dense", "sliding_window", "block_sparse"]:
        xs = [r["seq_len"] for r in rows if r["pattern"] == name]
        ax.plot(xs, [r["cells"] for r in rows if r["pattern"] == name], marker="o", label=name)

    ax.set_xlabel("sequence length")
    ax.set_ylabel("attended cells (mask.sum())")
    ax.set_title("theoretical: cells a gather-based implementation would need")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/plots/benchmark_theoretical.png")


if __name__ == "__main__":
    main()
