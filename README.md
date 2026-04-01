# Bridged Clustering

Reference code for Bridged Clustering (BC): a semi-supervised construction for learning cross-domain predictors from three sources of supervision:

- a small paired set `S`
- an input-only pool `X`
- an output-only pool `Y`

The central idea is simple. Cluster the two marginal spaces independently, use the paired set to recover a sparse bridge between clusters, and predict through the linked output cluster centroid.

## Repository Map

- [`bridged_clustering/core.py`](/Users/PatrickYe/Desktop/bridged_clustering/bridged_clustering/core.py)
  Core Bridged Clustering operations: splitting, balanced clustering, bridge estimation, centroid inference.
- [`bridged_clustering/text_pipeline.py`](/Users/PatrickYe/Desktop/bridged_clustering/bridged_clustering/text_pipeline.py)
  Shared orchestration for the image-text experiments.
- [`bridged_clustering/bioscan/`](/Users/PatrickYe/Desktop/bridged_clustering/bridged_clustering/bioscan/__init__.py)
  BIOSCAN-specific stack: family sampling, encoder loading, bridge helpers, experiment routines, and grid execution.
- [`bridged_clustering/datasets/`](/Users/PatrickYe/Desktop/bridged_clustering/bridged_clustering/datasets/__init__.py)
  Dataset preparation for Wikipedia, Flickr30k, and COCO.
- [`bridged_clustering/structures.py`](/Users/PatrickYe/Desktop/bridged_clustering/bridged_clustering/structures.py)
  Experiment specs, corpus descriptors, and model ordering.
- [`bridged_clustering/result_store.py`](/Users/PatrickYe/Desktop/bridged_clustering/bridged_clustering/result_store.py)
  Result buffers for the grid sweeps.
- [`baseline.py`](/Users/PatrickYe/Desktop/bridged_clustering/baseline.py)
  Baseline regressors used throughout the experiments.
- [`wiki.py`](/Users/PatrickYe/Desktop/bridged_clustering/wiki.py), [`flick.py`](/Users/PatrickYe/Desktop/bridged_clustering/flick.py), [`coco.py`](/Users/PatrickYe/Desktop/bridged_clustering/coco.py), [`bioscan.py`](/Users/PatrickYe/Desktop/bridged_clustering/bioscan.py)
  Dataset entry points and experiment grids.

## How The Text Stack Is Organized

The image-text experiments are split into four layers.

1. Dataset preparation
   Corpus-specific parsing, pruning, and candidate construction live under `bridged_clustering/datasets/`.
2. Algorithmic core
   The BC mechanics live in `bridged_clustering/core.py`.
3. Experiment orchestration
   Shared forward and reversed evaluation flows live in `bridged_clustering/text_pipeline.py`.
4. Script entry points
   The top-level scripts define the paper-style grids and write results.

That separation keeps the dataset logic close to the data, the clustering logic easy to audit, and the entry points short enough to read quickly.

The BIOSCAN path follows the same idea, but in its own package because it has a different vertical slice: image and DNA encoders, family-level sampling rules, and a separate forward/reversed evaluation stack.

## Running Experiments

Wikipedia:

```bash
python wiki.py --mode transductive
python wiki.py --mode inductive --reversed
```

Flickr30k:

```bash
python flick.py --mode transductive
python flick.py --mode inductive --reversed
```

COCO:

```bash
python coco.py --mode transductive
python coco.py --mode inductive --reversed
```

BIOSCAN:

```bash
python bioscan.py --mode transductive
python bioscan.py --mode inductive --reversed
```

Each run writes NumPy arrays for clustering quality and regression metrics into `results/<experiment_key>/`.

## Dependencies

The code assumes Python 3.10+ and the same library stack used in the experiments:

- `numpy`, `pandas`, `scikit-learn`
- `torch`, `torchvision`, `torch-geometric`
- `transformers`, `sentence-transformers`, `datasets`
- `k-means-constrained`
- `POT`
- `ortools`
- `matplotlib`, `tqdm`, `Pillow`

Large datasets are expected under `data/`. Generated outputs are written to `results/`.

## Reproducibility

- Grid sweeps use explicit seeds.
- Dataset preparation is deterministic given the sampled subset and seed.
- The text scripts keep dataset-specific hyperparameters in one place via experiment specs.
- Tooling configuration lives in [`pyproject.toml`](/Users/PatrickYe/Desktop/bridged_clustering/pyproject.toml), and generated artifacts are excluded via [`.gitignore`](/Users/PatrickYe/Desktop/bridged_clustering/.gitignore).
