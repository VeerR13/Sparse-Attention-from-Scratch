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
