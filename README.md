# sparse-attention-from-scratch

Dense and sparse self-attention implemented from first principles in PyTorch — no
`F.scaled_dot_product_attention`, no attention library — with a correctness harness,
memory/latency benchmarks, and a quality evaluation on a small character-level GPT.

Full attention compares every token against every other token, so its cost grows with the
**square** of sequence length. Sparse attention restricts which pairs are compared. This repo
implements both, verifies the sparse variants against a dense reference, and measures what the
speed and memory savings actually cost in model quality.

## Status

- [ ] 1.1 Manual dense attention (matmul → mask → softmax → matmul)
- [ ] 1.2 Sliding-window sparsity pattern
- [ ] 1.2 Block-sparse (BigBird-style: local + global + random)
- [ ] 1.3 Correctness harness: sparse matches dense on mutually-visible positions
- [ ] 1.4 NaN handling for fully-masked query rows
- [ ] 1.5 Benchmark: wall-clock + peak memory, sequence length 512 → 8192
- [ ] 1.6 Quality evaluation: 2-layer char-level GPT on TinyShakespeare
- [ ] 1.7 Writeup

## Repository structure

```
src/
  attention.py      # dense attention; takes an arbitrary boolean mask
  masks.py          # mask builders: causal, sliding-window, block-sparse
  model.py          # 2-layer character-level GPT
tests/
  test_attention.py # correctness harness (pytest)
scripts/
  benchmark.py      # wall-clock + peak memory vs. sequence length
  train.py          # TinyShakespeare training loop
results/
  plots/            # generated benchmark figures
README.md
WRITEUP.md
```

## Setup

```bash
git clone https://github.com/VeerR13/sparse-attention-from-scratch.git
cd sparse-attention-from-scratch
pip install -r requirements.txt
```

## Running

```bash
pytest tests/                      # correctness harness
python scripts/benchmark.py        # writes plots to results/plots/
python scripts/train.py --attn dense
python scripts/train.py --attn sliding_window
python scripts/train.py --attn block_sparse
```

## Design notes

The attention function takes the mask as an **argument** rather than constructing a causal mask
internally. Every sparsity pattern is therefore just a different boolean mask passed to the same
verified kernel, which keeps the dense implementation as the single source of truth and makes the
correctness harness a direct comparison.

## Benchmark results

Hardware: _TBD_

| Seq length | Dense (ms) | Sliding window (ms) | Block-sparse (ms) | Dense peak mem | Sliding peak mem | Block-sparse peak mem |
|-----------:|-----------:|--------------------:|------------------:|---------------:|-----------------:|----------------------:|
| 512        |            |                     |                   |                |                  |                       |
| 1024       |            |                     |                   |                |                  |                       |
| 2048       |            |                     |                   |                |                  |                       |
| 4096       |            |                     |                   |                |                  |                       |
| 8192       |            |                     |                   |                |                  |                       |

Numbers are relative and reproducible via `scripts/benchmark.py`.

## Quality results

2-layer character-level GPT, TinyShakespeare, identical hyperparameters across variants.

| Attention | Val loss |
|-----------|---------:|
| Dense | |
| Sliding window | |
| Block-sparse | |

## Findings

See [WRITEUP.md](WRITEUP.md) for the full discussion: what each pattern discards, why global
tokens carry disproportionate weight, and where sparsity did not pay for itself.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Beltagy et al., [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- Zaheer et al., [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
