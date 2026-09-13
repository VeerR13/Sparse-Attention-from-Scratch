# Notes

Raw notes as I hit things, to turn into WRITEUP.md later.

## 1.4 NaN handling

- Swept sliding_window over n in {16,32,64,100,128} x window in {1,2,4,8}, and
  block_sparse over n x block_size {4,8,16} x global {0,1,2} x random {0,1,3}.
  Zero empty rows in either pattern.
- Structural reason: both patterns always keep j=i - the window includes the
  token's own position, the block includes its own block. Not something I
  designed for, just a consequence of the natural way to build a window or a
  block.
- It becomes reachable the moment a pattern excludes the diagonal - a
  strictly-past window (j < i instead of j <= i) empties row 0 immediately.
- window_size=0 is the live reproduction case in my code right now.
- The most common real-world cause, which this project never hits: padding.
  Batching variable-length sequences gives padded rows nothing legitimate to
  attend to.
- The brief's "block boundaries" case: a block-sparse row whose only allowed
  blocks fall in the future gets emptied by the causal intersection. My global
  block 0 prevents it - every row can always see block 0.

## 1.5 benchmark

Masking applies after the matmul has already computed and allocated the full
n x n grid, so sparse semantics are achieved with zero saving in time or
memory - the measured benchmark shows dense, sliding-window, and block-sparse
as three overlapping curves, all following the same quadratic shape. Real
savings require never computing the skipped cells in the first place, which is
what a gather-based implementation would do, and is Task 4's whole premise.

## 1.6 quality eval

- Scoping choices, stated up front: 2 layers, single head (the brief only
  requires "2-layer", not multi-head - one head is enough to answer "does
  restricting attention hurt loss" and avoids reshaping entirely), context
  length 256 (big enough that the patterns actually restrict something).
- Matched average cells/row, not max lookback, per the earlier decision: dense
  ceiling is 128.5, sliding_window(76) achieves 64.87, block_sparse(16,1,2)
  achieves 65.5. Close but not exact - window_size is an integer, exact
  matching isn't possible, so I report achieved mean per run rather than
  claim a match.
- The block-sparse mean is not representative of a "typical" row. The last
  block gets full causal history (rows 241-256 see 241-256 cells) while a
  mid-sequence row sees roughly 65-80 - the last block alone is ~3x a typical
  row and pulls the mean up more than the block-to-block staircase does.
- Seed variance is not the same kind of thing for both patterns. Sliding
  window is deterministic - its only source of seed-to-seed variance is
  weight init and batch order. Block-sparse also has to redraw its random
  blocks per seed, which is why mean_cells_per_row itself moves slightly
  between seeds (65.5 at seed 0, 63.5 at seed 1, in a quick check). So
  block-sparse's spread across seeds answers a slightly different question
  than sliding-window's spread does.
- Reproducibility: block_sparse_mask is given its own generator, separate
  from the global RNG used for model init and batch order, specifically so
  that at a given seed, model init and the sequence of training batches are
  identical across all three patterns - the mask is the only thing that
  differs. Verified by asserting parameter count matches across every run.
