# Bridged Clustering

> Semi-supervised cross-modal prediction in the disjoint-data regime: a small paired set, two large unpaired marginals, and a cluster-level bridge learned from sparse supervision.

## Overview

This repository contains a research-oriented implementation of **Bridged Clustering (BC)** for cross-domain prediction when paired examples are scarce but unpaired observations from each modality are abundant.

The codebase supports four experiment families:

| Dataset family | Modalities | Canonical driver | Shared pipeline |
| --- | --- | --- | --- |
| COCO | image -> text, text -> image | `coco.py` | `bridged_clustering/text_pipeline.py` |
| Flickr30k | image -> text, text -> image | `flick.py` | `bridged_clustering/text_pipeline.py` |
| Wikipedia | image -> text, text -> image | `wiki.py` | `bridged_clustering/text_pipeline.py` |
| BIOSCAN | image -> DNA, DNA -> image | `bioscan.py` | `bridged_clustering/bioscan/` |

The implementation is organized around one reusable idea:

1. Split each semantic class into a small paired subset and two unmatched pools.
2. Cluster the two modality marginals independently with size-constrained KMeans.
3. Learn a **discrete bridge** from input clusters to output clusters using only the paired examples.
4. Predict in the output space by routing a test point through its inferred cluster and an output-space centroid or prototype.

This makes the repository useful both as:

- a **reference implementation** of the BC pipeline, and
- a **reproducible experiment harness** that writes paper-style result tensors consumed by the analysis notebooks.

## Why This Repository Is Structured This Way

Research code often has two competing requirements:

- it must preserve the semantics of the original experiments closely enough to reproduce result tables and plots;
- it must also be legible enough that another researcher can inspect, test, and repurpose individual components.

This codebase tries to satisfy both:

- the **algorithmic core** is isolated in reusable functions;
- dataset-specific quirks live in dataset adapters and experiment wrappers;
- the canonical drivers still emit the legacy `npy` tensors expected by downstream notebooks;
- synthetic end-to-end tests exercise all four experiment families without requiring the canonical datasets.

## Method Summary

Let:

- `S` be a small set of paired samples `(x, y)`,
- `X` be an input-only pool,
- `Y` be an output-only pool.

Bridged Clustering assumes the two modalities share latent semantic structure even when most observations are unpaired.

### Forward direction

For an input-to-output experiment:

1. Form an **input pool** by combining `X` with the input side of `S`.
2. Form an **output pool** by combining `Y` with the output side of `S`.
3. Fit balanced clustering independently in each pool.
4. Use the paired examples in `S` to estimate a majority-vote mapping
   `input cluster -> output cluster`.
5. Assign each test input to its nearest input centroid, map it across the bridge, and emit the corresponding output centroid or prototype.

### Reverse direction

The reverse experiments reuse the same idea with the roles of the modalities swapped. This is implemented explicitly for both the text benchmarks and the BIOSCAN image-DNA setting.

### Evaluation

The canonical experiment scripts record:

- `AMI_X`: clustering quality in the input modality,
- `AMI_Y`: clustering quality in the output modality,
- `accuracy`: bridge accuracy against an oracle cluster correspondence,
- `mae.npy` and `mse.npy`: regression losses for BC and the baseline suite.

## Repository Map

### Core algorithm

- `bridged_clustering/core.py`
  The main algorithmic entry point. This module implements split construction, balanced clustering, bridge estimation, oracle construction, centroid-based inference, and clustering diagnostics.

### Shared image-text stack

- `bridged_clustering/text_pipeline.py`
  Shared runner for COCO, Flickr30k, and Wikipedia. It executes the BC forward and reverse pipelines, evaluates baselines, and returns the metric structure consumed by the drivers.
- `bridged_clustering/text.py`
  Text-specific helpers such as nearest-text recovery, candidate-aligned evaluation, and wrapping numeric regressors for text prediction.
- `bridged_clustering/datasets/coco.py`
- `bridged_clustering/datasets/flickr.py`
- `bridged_clustering/datasets/wiki.py`
  Dataset preparation layers that convert raw dataset files into the common `PreparedTextCorpus` structure.

### BIOSCAN stack

- `bridged_clustering/bioscan/grid.py`
  Canonical BIOSCAN grid runner behind `python bioscan.py ...`.
- `bridged_clustering/bioscan/experiments.py`
  Forward and reversed BIOSCAN experiments.
- `bridged_clustering/bioscan/data.py`
  Family sampling and split logic.
- `bridged_clustering/bioscan/encoders.py`
  Image encoder and BarcodeBERT loading plus feature extraction.
- `bridged_clustering/bioscan/bridge.py`
  BIOSCAN-specific bridge construction and inference utilities.

### Outputs and result storage

- `bridged_clustering/result_store.py`
  Metric storage for the image-text grids.
- `bridged_clustering/bioscan/results.py`
  BIOSCAN-specific metric storage preserving the expected tensor layout.
- `utils/plotting.ipynb`
  Notebook for consuming the saved result tensors.

### Experiment entrypoints

- `coco.py`
- `flick.py`
- `wiki.py`
- `bioscan.py`

These are the canonical CLI entrypoints for the four experiment families.

### Tests

- `tests/test_bioscan_data.py`
- `tests/test_bioscan_experiments.py`
- `tests/test_text_experiment_drivers.py`

The test suite uses synthetic data and lightweight stubs to verify that every experiment family can complete a run and write the expected outputs without access to the canonical datasets.

## Installation

The code assumes **Python 3.10+**.

### Minimal setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Main dependencies

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `torch`, `torchvision`, `torch-geometric`
- `transformers`, `sentence-transformers`, `datasets`
- `k-means-constrained`, `ortools`
- `adapt`, `POT`
- `pyarrow`, `Pillow`, `matplotlib`, `tqdm`

Some experiment paths depend on heavyweight libraries that may not be available in a minimal environment. The code now raises clearer dependency errors when such components are missing.

## Data Layout

Raw data are expected under `data/`. Generated outputs are written to `results/`.

The repository assumes dataset-specific source files such as:

- `data/coco_2.csv`
- `data/flickr30k.parquet`
- `data/wiki_df.csv`
- `data/bioscan_data.csv`
- `data/bioscan_images/`

The dataset loaders in `bridged_clustering/datasets/` and `bridged_clustering/bioscan/data.py` document the expected structure more precisely through the columns they consume.

## Running The Canonical Experiments

### Wikipedia

```bash
python wiki.py --mode transductive
python wiki.py --mode inductive --reversed
```

### Flickr30k

```bash
python flick.py --mode transductive
python flick.py --mode inductive --reversed
```

### COCO

```bash
python coco.py --mode transductive
python coco.py --mode inductive --reversed
```

### BIOSCAN

```bash
python bioscan.py --mode transductive
python bioscan.py --mode inductive --reversed
```

For BIOSCAN, the DNA encoder uses `bioscan-ml/BarcodeBERT`. Depending on the local Hugging Face state, loading may require an explicit remote-code opt-in:

```bash
BRIDGED_CLUSTERING_ALLOW_REMOTE_CODE=1 python bioscan.py --mode transductive
```

## Output Convention

Each canonical run writes NumPy tensors to `results/<experiment_key>/`.

The standard files are:

- `ami_x.npy`
- `ami_y.npy`
- `accuracy.npy`
- `mae.npy`
- `mse.npy`

This mirrors the output format used by the original analysis workflow and keeps the notebook-side plotting code simple.

## Baselines

The repository evaluates Bridged Clustering against a family of supervised, semi-supervised, and transport-based baselines.

For the image-text benchmarks, the canonical order is:

- `BKM`
- `KNN`
- `FixMatch`
- `Laplacian RLS`
- `TSVR`
- `TNNR`
- `UCVME`
- `GCN`
- `KMM`
- `EM`
- `EOT`
- `GW`

For BIOSCAN, the same ordering is used except that `KMM` is recorded under the fuller label `Kernel Mean Matching`.

The baseline implementations live behind `baseline.py`, while the experiment stacks call them through thin wrappers so that the pipelines remain testable in dependency-light environments.

## Reproducibility Notes

- Randomness is explicitly seeded across the grid runners.
- The drivers preserve the paper-style sweep structure and save intermediate results in the legacy tensor format.
- BIOSCAN loads the expensive encoders once per grid run and reuses them across trials.
- The text drivers now expose callable grid helpers, which makes the experiment definitions easier to test and inspect programmatically.

## Testing

The repository includes synthetic end-to-end tests that avoid the canonical datasets while still exercising the full experiment wiring:

```bash
pytest tests/test_bioscan_data.py tests/test_bioscan_experiments.py tests/test_text_experiment_drivers.py
```

These tests cover:

- BIOSCAN family sampling and split construction,
- forward and reversed BIOSCAN execution on synthetic features,
- forward and reversed COCO/Flickr30k/Wikipedia grids on synthetic corpora,
- result tensor creation and expected output shapes.

## Practical Caveats

- `k-means-constrained` is required for the balanced clustering assumption used throughout the method.
- `torch-geometric` and some transport dependencies are only needed for the heavier baseline stack.
- BIOSCAN is the most operationally demanding experiment family because it couples image features, DNA embeddings, and dataset-specific file layout assumptions.
- Large runs can be expensive in both memory and wall-clock time, especially for the full paper-style sweeps.

## If You Are Reviewing The Code

The fastest path to the main ideas is:

1. Read `bridged_clustering/core.py`.
2. Inspect `bridged_clustering/text_pipeline.py` for the shared benchmark logic.
3. Compare one text driver such as `coco.py` with `bridged_clustering/bioscan/grid.py` to see how the shared and dataset-specific layers differ.
4. Run the synthetic test suite before attempting a full data-dependent experiment.

That path should give you a clear view of:

- what is generic to Bridged Clustering,
- what is specific to a dataset family,
- how experiment sweeps are parameterized,
- and how outputs are serialized for downstream analysis.
