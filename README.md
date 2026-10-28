# How Far Can You Push a Frozen Backbone?

Layer-wise linear probes of an ImageNet ResNet-50 across five image domains.
COMS4036A/COMS7050A Computer Vision project: Mashego Mabeloane (2654606), Basimane Ravele Lebopa (2674912).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USER/YOUR-REPO/blob/main/resnet50_transfer_project.ipynb)

## Contents
- `resnet50_transfer_project.ipynb`: the whole pipeline (data, probes, end-to-end training, tables, figures)
- `requirements.txt`: Python packages

## Setup
Recommended: open the notebook in Colab with a GPU runtime (Runtime ▸ Change runtime type ▸ T4 GPU).

Locally (Python 3.10+, CUDA GPU strongly recommended):

    pip install -r requirements.txt
    jupyter notebook resnet50_transfer_project.ipynb

## Data
| Dataset | Source | How it is obtained |
|---|---|---|
| Oxford-IIIT Pets | robots.ox.ac.uk | downloaded automatically by the notebook |
| GTSRB | torchvision | downloaded automatically |
| Satellite (4-class) | Kaggle `mahmoudreda55/satellite-image-classification` | Kaggle API |
| C-NMC Leukemia | Kaggle `andrewmvd/leukemia-classification` | Kaggle API |
| Chest X-ray Pneumonia | Kaggle `paultimothymooney/chest-xray-pneumonia` | Kaggle API |

The Kaggle downloads need an API token. In Colab, add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab Secrets
(key icon in the left sidebar). Locally, put `kaggle.json` in `~/.kaggle/`.

Splits: official test sets for Pets and GTSRB; GTSRB validation split by track; C-NMC and Chest X-ray split
by patient; satellite stratified 70/10/20. Each split is capped at 4000/1000/2000 images by stratified
subsampling (`MAX_TRAIN`, `MAX_VAL`, `MAX_TEST` in the config cell).

## Running
Run all cells in order (Runtime ▸ Run all). Section 6 runs the four conditions on all five datasets
(about 15–30 minutes on a T4) and writes `results/results.csv` after each dataset; if Colab disconnects,
re-run Section 6 and it continues from the next unfinished dataset. Section 7 rebuilds the tables and
figures from the CSV, so the report's numbers can be checked against the committed CSVs without retraining.

## Reproducibility
All randomness is seeded with `SEED = 0` (Python, NumPy, PyTorch, CUDA, data splits, probe initialisation).
GPU kernels are not fully deterministic, so a re-run may differ by about ±0.01 macro-F1. The committed
CSVs are the exact runs reported.


