# Writeup — Sparse Attention from Scratch

## What this is

I implemented dense self-attention by hand in PyTorch, then built two sparsity patterns on top
of it — a sliding window and a BigBird-style block-sparse pattern — and measured what each one
costs. The repo has a correctness harness, NaN handling, a benchmark across sequence lengths
512 to 8192, and a quality comparison on a small character-level GPT trained on TinyShakespeare.

The short version of what I found: masking gives you sparse behaviour with none of the savings,
halving the information a token can see cost almost nothing on this task, and the block-sparse
pattern unexpectedly ended up slightly ahead of dense.

## Why I picked this task

Task 1 looked like the most basic of the four and like the right starting point. My thinking was
that building the simple version myself would give me an understanding I could then apply my own
way, rather than starting from someone else's more complicated solution to the same problem and
working backwards from it.

## Attention, and one structural decision

`dense_attention` is four operations: compute a score for every query-key pair with a matmul,
apply the mask, softmax each row into weights that sum to 1, then blend the value vectors by those
weights. The score grid is `seq_len × seq_len`, which is where the quadratic cost comes from.

The function takes the mask as an **argument** rather than constructing a causal mask internally,
which is what most implementations of this do. Partly that was to keep things simple — with very
little coding experience, one function with one job is less to get wrong than several near-copies.
But it turned out to matter more than that for the rest of the project. Because every pattern is
just a different boolean grid passed to the same function, all three run through one implementation
that has been checked against a reference. When dense and sparse outputs differ, the difference
must have come from the mask, because there is no other code path it could have come from. If I
had written three separate attention functions instead, a difference in results could have meant
the pattern was worse or that one of the three implementations had a bug, and I would have had no
way to tell which.

I also checked my dense attention against `F.scaled_dot_product_attention` and it matches to 1e-6.
The task forbids using that function as the implementation, but item 1.1 calls dense attention
"your reference", which means every later correctness claim inherits its bugs. My harness compares
sparse against dense — if dense were wrong, they would agree perfectly and every test would pass
on a broken baseline. That one comparison is what anchors the rest of the work to something
external. It's used only in `tests/`, with a comment saying why.

## The two patterns

Sliding window was straightforward, and I understood it first: token `i` can see the `window_size`
positions behind it and nothing older. It is a diagonal band, and each row has a fixed width no
matter how long the sequence is, which is where the saving would come from.

Block-sparse was harder to get right. The pattern is three rules combined: a local window over neighbouring blocks, global
blocks that everything can see, and a few random blocks per row. What made it click was
understanding that the decisions are made per *block-pair* and only then expanded to tokens — not
per token. That matters because GPUs read contiguous memory quickly and scattered memory slowly,
so sparsity that saves arithmetic but scatters the surviving work gains nothing in practice.

Both patterns are causal by construction rather than built bidirectionally and then intersected
with a causal mask. The only thing consuming these masks is a causal model, so building them
bidirectionally would solve a problem I don't have. It would also introduce a real one: if random
blocks are sampled from the whole range and the illegal ones filtered out afterwards, early
positions lose nearly all their picks while late positions lose about half — a systematic bias
against exactly the positions that already have the least context available. Sampling only from
causally legal blocks avoids that.

## What each pattern discards

Sliding window discards everything beyond the window. On this task that cost almost nothing:
averaged over three seeds it came out at 1.7548 validation loss against dense's 1.7573, and the
sign of the difference actually flipped on one seed. Predicting the next *character* is a local
job — you rarely need to look back 128 characters to guess the next letter. A window of 76 covers
essentially everything that matters here. It would break on a task that genuinely needs long-range
recall, like answering a question about something stated much earlier in a document.

Block-sparse discards differently. At a matched budget (65.5 cells per row against sliding
window's 64.87), it gives up the guarantee of a constant lookback. Because the window moves in
whole blocks, the effective lookback resets at every block boundary and ramps up across the block —
cells per row goes 5, 6, 7, 8, then back to 5. A token at the start of its block sees noticeably
less than one at the end. There's also a much larger source of unevenness: the last block gets full
attention, so at sequence length 1024 it sees 1024 cells against a typical row's ~350.

## Why global tokens matter disproportionately

This is the clearest result in the project, and it comes from the two sparse patterns having
nearly the same information budget but different shapes.

Sliding window is purely local and it matched dense. Block-sparse, which is a local window plus
global blocks plus random blocks at the same budget, beat dense by 0.0186 on average and was ahead
in all three seeds. It also beat sliding window in all three seeds, by 0.0161. Since the only
structural difference between my two sparse patterns is the global and random connections,
whatever is being gained has to be coming from those.

There's a connectivity argument for why. With a window of width `w` and `L` layers, information can
travel at most `L × w` positions. My model has 2 layers and a window of 76, so two tokens more than
about 152 apart can never influence each other at all — not weakly, but structurally never. A
single global block fixes that for one row and one column of the mask: everyone can reach it, so
any two tokens become two hops apart regardless of how far apart they are.

One thing I found that isn't in the paper: **BigBird's two-way global attention collapses to
one-way under causal masking.** For the first block, "sees everyone" is a no-op, because nothing
precedes it — but "everyone sees it" survives completely, which gives the attention-sink behaviour.
For the last block it's the mirror image. BigBird's symmetry property, that if A attends to B then
B attends to A, only holds in the bidirectional encoder setting it was published for. In a causal
decoder you get half of it on each end.

## The benchmark: sparse semantics, no sparse savings

Measured on a Colab T4, forward pass only, masks built outside the timed region, median of 8
repeats after a warmup.

Peak memory at sequence length 8192 was **1155.67 MB for all three patterns, identical**. Not
close — identical to three decimal places, at every sequence length I tested.

The reason is the order of operations. The matmul runs first and computes all `n × n` scores,
allocating the full grid. The mask arrives afterwards and overwrites entries with `-inf`. Nothing
is ever skipped; the work is already done and the memory already allocated by the time the mask has
any say. Wall-clock agreed with this — the three patterns were within about 1% of each other at
every size above 512.

So I also counted how many cells each mask actually allows, which is what a gather-based
implementation would compute. At 8192 the sliding window needs 2,064,512 cells against dense's
33,558,528 — 6.2%. At that ratio a gather-based implementation would need roughly 71 MB rather than
the 1156 MB I measured. **About 16× of memory is being paid for and thrown away.**

Two things I don't trust in my own numbers. At sequence length 512 the sparse patterns look about
19% faster; the whole forward pass is 0.4 ms there, small enough that kernel launch overhead
dominates, so I don't believe it. And the memory scaling ratios start at ×2.03 and only reach ×3.89
by 8192, short of the ×4 you'd expect from a purely quadratic term, because at small sizes the
q/k/v tensors and allocator overhead are a meaningful fraction of the total.

Building the slow version was still worth it. It answers the quality question exactly — a mask-based
sliding window produces bit-identical output to an optimised kernel, so you can evaluate whether a
pattern is worth having before writing one. And it's the oracle you would check a fast kernel
against, which is exactly the role item 1.1 gives it.

## Quality evaluation

2-layer single-head character-level GPT, 446k parameters, context length 256, 5000 steps, three
seeds per pattern. Within a seed, all three patterns get identical initialisation and identical
batch order — the mask is the only thing that differs.

| Pattern | Mean cells/row | Val loss, 3 seeds | Mean |
|---|---:|---|---:|
| Dense | 128.5 | 1.7351 / 1.7364 / 1.8005 | 1.7573 |
| Sliding window | 64.87 | 1.7302 / 1.7320 / 1.8022 | 1.7548 |
| Block-sparse | 63.5–66.5 | 1.7151 / 1.7112 / 1.7898 | 1.7387 |

The most important thing about this table is that the seed effect is larger than the pattern
effect. Dense on its own ranges from 1.7351 to 1.8005 — a spread of 0.065 — while the gap between
dense and block-sparse is 0.019, about 3.5× smaller. Seed 2 was simply a worse run for all three
patterns. If I had compared one dense run against one sparse run without pairing them, the result
would have been unreadable noise. The paired design is the only reason there is anything to
conclude here at all.

Compared per seed rather than by average:

```
sliding_window vs dense:   -0.0049, -0.0044, +0.0017    sign flips, so no effect
block_sparse   vs dense:   -0.0200, -0.0252, -0.0107    consistent
block_sparse   vs sliding: -0.0151, -0.0208, -0.0124    consistent
```

### Sparse beat dense, which I did not expect

I had assumed this couldn't legitimately happen, since dense has strictly more information
available. My training script prints a warning if a sparse pattern wins, and the warning fired.

I checked the obvious explanations. The mask does reach the model — with a 64-wide window, outputs
are identical to dense for the first 64 positions and start diverging at exactly position 64, which
is where they should. Initialisation and batch order are identical across patterns within a seed.
Evaluation uses the same mask as training. I did not find a bug.

What I think is happening is that the gain comes from global tokens rather than from sparsity.
Sliding window, which is purely local, matched dense. Block-sparse, which adds global and random
connections at the same budget, beat it. A global block hands the model a routing hub for free;
dense has to learn to build one out of its own attention weights.

The loss curves support reading this as faster convergence rather than greater capability. The gap
is around 0.1 near step 1000 and about 0.02 by step 5000 — it is narrowing, not widening. Validation
loss was still falling at step 4999 for every pattern, so none of these models is near convergence,
and dense's information advantage may simply not have had time to pay off.

I can't claim this is significant. Three seeds all pointing the same way is a sign test p-value of
0.125, and the effect is smaller than the seed-to-seed spread. The honest statement is that
block-sparse was consistently ahead at this training budget, most plausibly because of global
tokens, and that a longer run would be needed to know whether dense eventually overtakes it.

## Things that went wrong

**A slicing bug that silently inverted a control condition.** `block_mask[-num_global_blocks:, :]`
with `num_global_blocks = 0` selects the entire array rather than nothing, because `-0 == 0` in a
slice. This meant a "local window only" configuration was actually testing "every block is global",
and nothing about it looked wrong from the outside. It's now behind an explicit `> 0` guard with a
comment, because the failure mode is invisible.

**My correctness harness is weaker than it first appears.** It compares `dense_attention` against
`dense_attention` with two different masks, so a bug inside that function would affect both sides
identically and cancel out. It verifies that the two masks agree with each other, not that attention
is correct — correctness rests entirely on the separate comparison against PyTorch's kernel. Adding
one ground-truth assertion helped: under a causal mask, row 0 can only see itself, so its output
must exactly equal the first value vector, which is checkable without any reference implementation.
Coverage is also narrower than it looks, since rows where the two masks are identical are the ones
checked directly — 4 of 16 rows for sliding window, 16 of 32 for block-sparse.

**The NaN case is unreachable in my patterns,** which surprised me. I expected to hit
`softmax(-inf, -inf, ...)` naturally at some point, since the task describes it as something that
happens for real. It can't happen here: both patterns always keep the diagonal, so no row is ever
fully masked. That wasn't a deliberate defence — it just follows from the natural way to build a
window or a block. `sliding_window_mask(n, 0)` is the one configuration that reproduces it, and the
most common real cause, padded sequences in a batch, never arises in this project.

## Limitations and what I'd do next

There is no gather-based implementation, so no speed or memory saving was demonstrated and none was
possible — everything in the benchmark section follows from that. The quality runs are undertrained
at 5000 steps with validation loss still falling. Three seeds is not many for an effect this small.
Random blocks are sampled from all causally legal blocks including ones the window already covers,
so `num_random_blocks` slightly overstates the number of genuinely new connections. And the model
is single-head with layer norm after the residual add, both chosen to keep the scope manageable.

With more time the first thing I would do is rerun dense and block-sparse for 10,000 steps instead
of 5,000. My own loss curves suggest dense might overtake, and that would turn "block-sparse wins"
into "block-sparse converges faster but dense wins asymptotically" — a better-supported and more
interesting claim than either result on its own. After that, a gather-based sliding window, which
is the more tractable of the two patterns to implement that way, to see the memory curve actually
flatten.
