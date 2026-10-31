# Single-cell neighborhood fusion

Code archive accompanying *Evaluating Neighborhood Fusion and Clustering Readouts in Single-Cell Graph Contrastive Learning*.

## Contents

`code.zip` contains the preserved Python experiment sources and protocol metadata. Extract it to recover the original directory structure. `SOURCE_MANIFEST.json` records SHA-256 hashes of every archived source. Data, cell labels, embeddings, checkpoints, runtime logs and scored outputs are excluded.

Main entry points inside `nb_scgaam_comparison/experiments/`:

- `cross_dataset_v1/`: original Fusion training, preparation and readouts.
- `review_experiments_20261030/worker.py`: density, random, PCA, MSE and no-auxiliary controls.
- `review_experiments_20261030/scsamac_worker.py`: consistently stabilized scSAMAC evaluation.
- `review_experiments_20261030/evaluate.py`: paired evaluation.
- `review_experiments_20261030/structural_diagnostics.py`: graph and class diagnostics.

## Execution scope

This is an archival source release, not a verified one-command installation. The recorded experiments used Python 3.13.9, PyTorch 2.9.1+cu128, NVIDIA RTX 5070 Ti, float64 count pretraining and float32 graph training. NumPy, SciPy, scikit-learn, h5py, pandas, scanpy, torch-geometric and threadpoolctl are among the dependencies. Some archived orchestration scripts retain original Windows paths and require the prepared inputs and prerequisite runs described by their protocols. Downloading source data alone does not create these intermediate inputs. Full clean-machine reproduction has not yet been verified.

## Data

Data are not redistributed. Download locations and source versions are listed in the manuscript supplement. Development data: https://zenodo.org/records/15272789 . Primary data: https://github.com/xuebaliang/scziDesk/tree/fc719e7086381d6e7713870b52cee44b015e5762/dataset . Muraro original study: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241 .

## Attribution

The implementation includes adaptations of scDeepCluster and scGAAM, and comparison adapters for external methods. Consult source notices and the paper citations. No blanket relicensing of third-party code is asserted by this archive.
