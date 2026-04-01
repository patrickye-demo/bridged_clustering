# Bridged Clustering

Bridged Clustering (BC) is a semi-supervised method for cross-domain prediction with a small paired set `S`, an input-only pool `X`, and an output-only pool `Y`.
It targets the disjoint-data regime where most inputs and outputs are unpaired but still share latent semantic structure.
The key idea is to cluster the two marginal spaces independently and use the paired subset only to learn a sparse bridge between clusters.

## Start Here

- [`bridged_clustering/core.py`]
  Inspect the method itself: split construction, balanced clustering, bridge estimation, oracle comparison, and centroid-based inference.
- [`bridged_clustering/text_pipeline.py`]
  Shared image-text experiment runner used by Wikipedia, Flickr30k, and COCO.
- [`bridged_clustering/bioscan/`]
  BIOSCAN-specific stack for the image-DNA setting.
- [`wiki.py`], [`flick.py`], [`coco.py`], [`bioscan.py`]
  Canonical experiment drivers that write the legacy result tensors consumed by [`utils/plotting.ipynb`].

## Method Overview

- Split each semantic group into supervised pairs, unmatched input-only examples, unmatched output-only examples, and evaluation examples.
- Fit size-constrained KMeans independently in the input and output spaces.
- Estimate a discrete bridge from input clusters to output clusters using only the paired subset.
- Predict by routing each test input through its input cluster to an output centroid or prototype.

## Canonical Commands

After `pip install -r requirements.txt`, the main entrypoints are:

```bash
python wiki.py --mode transductive
python coco.py --mode inductive --reversed
BRIDGED_CLUSTERING_ALLOW_REMOTE_CODE=1 python bioscan.py --mode transductive
```

Each run writes `ami_x.npy`, `ami_y.npy`, `accuracy.npy`, `mae.npy`, and `mse.npy` into `results/<experiment_key>/`.

## Dependencies

The code assumes Python 3.10+.

The main dependencies in `requirements.txt` are:

- `numpy`, `pandas`, `scikit-learn`
- `torch`, `torchvision`, `torch-geometric`
- `transformers`, `sentence-transformers`, `datasets`
- `adapt`
- `k-means-constrained`
- `POT`
- `ortools`
- `matplotlib`, `tqdm`, `Pillow`
- `pyarrow`

Large datasets are expected under `data/`. Generated outputs are written to `results/`.

For BIOSCAN, `bioscan-ml/BarcodeBERT` may require remote-code loading from Hugging Face. The loader requires an explicit opt-in:

```bash
BRIDGED_CLUSTERING_ALLOW_REMOTE_CODE=1 python bioscan.py --mode transductive
```