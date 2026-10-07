# Paper claims verification

This directory contains a standalone script that recomputes the numbers reported
in the paper *Prior or Feedback? What an LLM Uses When Adapting Neural Operators*
(accepted to the NeurIPS 2026 AI4Science Workshop) from the released run records.

## Files

- `verify_claims.py` — recomputes endpoint medians, win counts, first-proposal
  statistics, cold-start LR-prior effects, score-reassignment replay effects
  and their controls, the TPE ten-startup control, and the supplementary
  search-trajectory and early-budget tables.
- `expected.json` — paper values, tolerances, and source locations for each
  checked claim. The script exits with status 1 if any claim in this file is
  outside tolerance, if any computation errors, or if a checked claim has no
  computed result.

## Requirements

- Python 3.10 or newer.
- NumPy, installed through the `paper` optional dependency group.

No PyTorch, PDEBench, FNO training code, or LLM API calls are needed.

## Run

1. Download the run records from the dataset record (DOI: [10.5281/zenodo.23213042](https://doi.org/10.5281/zenodo.23213042)) and unzip
them to a directory of your choice, for example `records/`.

2. From the exported repository root:

```bash
uv sync --frozen --extra paper
uv run --frozen --extra paper python paper/verify_claims.py --records records
```

With a system Python that already has NumPy:

```bash
python3 paper/verify_claims.py --records records
```

To use a different expected-values file:

```bash
uv run --frozen --extra paper python paper/verify_claims.py \
  --records records --expected paper/expected.json
```

## What is verified

- **Endpoint Table 2** (main text): per-cell median final test nRMSE for the
  LLM controller, Random, and TPE across the 12 adaptation cells.
- **Win counts**: LLM controller versus Random (`36/36`) and versus TPE
  (`35/36`).
- **First-proposal quality** (supplement Table 3): per-cell, per-seed
  percentage of the random pool matched or beaten by the first LLM proposal,
  plus the pooled median and threshold counts.
- **Cold-start LR prior** (main text and supplement): median base learning
  rates and shares of `0.001` proposals under advection-text, burgers-text,
  and no-text descriptions, the Mann-Whitney U test, and the aggregate
  action-distance description-separation statistic.
- **Score-reassignment replay** (main text and supplement): within-family and
  cross-PDE reassign versus notation/resampling medians, cell-clustered
  sign-flip p-values, the hierarchical bootstrap CI for the within-family
  reassignment effect, the independent within-family feedback replay, the
  cross-family feedback bundle, and relabelled-best diagnostics.
- **TPE ten-startup control** (main text appendix): LLM wins and median
  TPE-to-LLM ratios for the within-family and cross-family control cells.
- **Search trajectories** (supplement Table 5): median consecutive action
  distances by trial window, per-policy Spearman rho, and counts of declining
  trajectories.
- **Early-budget timing** (supplement Table 4): LLM best after 1/3/5 trials
  versus Random/TPE best after 20 trials.

## Notes on implementation

- `action_distance` implements the paper's 12-coordinate mean absolute
  difference: normalised continuous coordinates plus categorical Hamming
  distance.
- The within-family replay uses the notation control (`sham` arm) as the
  baseline for the main reassignment effect, matching the registered
  hierarchical bootstrap.
- The cold-start description-separation statistic preserves the producer's
  CSV row order and the seeded 20,000-draw label shuffle (seed 0).

## Linting

From the exported repository root:

```bash
uv run --frozen --extra paper --group dev ruff check paper/
uv run --frozen --extra paper --group dev ruff format --check paper/
```
