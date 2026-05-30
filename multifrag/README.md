# multifrag — multi-fragment voting (prototype)

Phase 1's per-fragment coarse classifier plateaus at **~88%**. The residual
error is concentrated in `text` ↔ `archive` and is largely **irreducible per
fragment** — a single 4096-byte fragment of a text file and a fragment of a
text file stored inside a tar are byte-identical.

The way past that ceiling is **file-level context**: one fragment is ambiguous,
but a file produces many fragments. Averaging the Phase-1 model's predictions
across a file's fragments should beat any single-fragment prediction — errors
on individual fragments cancel out.

## Smoke test

`smoke_vote_sim.py` runs the trained Phase-1 model over the val split and
measures the voting payoff. It needs no per-file metadata:

1. **Run-length check + real-run voting** — if fragments from one file happen to
   be stored consecutively, each same-label run is voted directly (an honest
   number). If the order is shuffled, real grouping would need a dataset regen.
2. **Simulated voting curve** — partitions each group's fragments into chunks of
   K, mean-averages their predicted probabilities, and scores the voted argmax.
   Reports voted accuracy as K grows, overall and per group.

Caveat: simulated chunks mix fragments from different files, so their errors are
more independent than one real file's fragments would be — the curve is an
**optimistic upper bound**. Flat ⇒ voting won't help; steep ⇒ worth building
with real per-file grouping.

```
PYTHONPATH=src python multifrag/smoke_vote_sim.py \
    --binary-dir data/4k_1/binary \
    --checkpoint checkpoints/phase1_archive/best.pt
```

Runtime ≈ inference over the val split (a few minutes); use `--max-samples` to
cap it. Reads the `best_source` field so it loads the EMA or raw weights to
match `best.pt`.

## Files

| file | purpose |
|---|---|
| `smoke_vote_sim.py` | runs Phase-1 inference + the voting smoke test |

## What the result decides

- **Steep voting curve** → build real multi-fragment voting: needs a per-fragment
  file-id (group fragments by file at inference). That's the next step.
- **Flat curve** → voting can't rescue the ceiling either; the error is
  per-file-irreducible and the taxonomy (tar in `archive`) is the real limit.
