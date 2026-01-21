# YoloTest

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-check_LICENSE-orange.svg)]()
[![Status](https://img.shields.io/badge/status-experimental-yellow.svg)]()

> A compact PyTorch reimplementation and learning playground of the YOLO v1 object detection model (for experimentation, education, and small-scale training).

Table of contents
- [What the project does](#what-the-project-does)
- [Why this project is useful](#why-this-project-is-useful)
- [Quick start](#quick-start)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Prepare dataset](#prepare-dataset)
  - [Run training](#run-training)
  - [Run inference / evaluate](#run-inference--evaluate)
- [Project structure](#project-structure)
- [Dataset / label format](#dataset--label-format)
- [Where to get help](#where-to-get-help)
- [Maintainers & contributing](#maintainers--contributing)
- [License](#license)

## What the project does

YoloTest is an educational implementation of the YOLO v1 object detection architecture in PyTorch. It contains:
- a compact Darknet-like CNN and fully-connected detection head (model.py)
- a custom dataset loader for VOC-style examples (dataset.py)
- training loop and training utilities (train.py, utils.py)
- the YOLO loss implementation (loss.py)
- helper functions for IOU, NMS, mAP, converting predictions, plotting, and checkpointing (utils.py)

The code expects images resized to 448×448 and implements S=7 grid, B=2 boxes, and up to C=20 classes by default.

## Why this project is useful

- Educational: a smaller, readable implementation of YOLO v1 for learning how the architecture, loss, and post-processing work.
- Practical playground: easy to tweak architecture, loss weighting, and training hyperparameters.
- Useful utilities: IOU, NMS, mAP calculation, and visualization functions to inspect intermediate outputs.

## Quick start

### Requirements
Minimum (approximate):
- Python 3.8+
- PyTorch (compatible with your CUDA or CPU): torch
- torchvision
- pandas
- pillow (PIL)
- numpy
- matplotlib
- tqdm

Install with pip (example — pick the correct torch wheel for your environment):
pip install torch torchvision pandas pillow numpy matplotlib tqdm



### Install
1. Clone the repo:
   git clone https://github.com/noop-rabbit/YoloTest.git
   cd YoloTest

2. Create a virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   # or install packages manually:
   pip install torch torchvision pandas pillow numpy matplotlib tqdm

### Prepare dataset
The code uses a simple CSV listing images and corresponding label files. Default paths in train.py:
- IMG_DIR = `data/images`
- LABEL_DIR = `data/labels`
- Training CSV example: `data/100examples.csv`
- Test CSV example: `data/test.csv`

CSV layout (no header, per-row):
- column 0: image filename (relative to IMG_DIR)
- column 1: label filename (relative to LABEL_DIR)

Label file format (one object per line, space-separated):
class_index x_center y_center width height

- class_index: integer class id (0 .. C-1)
- x_center, y_center, width, height: floats in normalized image coordinates (0..1)
- Example label file (one object):
  0 0.512 0.432 0.234 0.418

### Run training
The main training entrypoint is `train.py`. Default hyperparameters are defined near the top of `train.py`:

- LEARNING_RATE = 2e-5
- EPOCHS = 100
- BATCH_SIZE = 32
- IMG_DIR, LABEL_DIR and CSV filenames also defined in `train.py`

To start training:
python train.py

Notes:
- The script uses GPU when available (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`).
- To load a checkpoint before training, toggle `LOAD_MODEL` and set `LOAD_MODEL_FILE` in `train.py`.
- Adjust hyperparameters in `train.py` as needed.

### Run inference / evaluate
Utilities in `utils.py` support converting cell predictions into bounding boxes, running non-maximum suppression and computing mAP.

Typical usage pattern (examples are in code and are simple to reuse):

- Use `get_bboxes(loader, model, iou_threshold, threshold)` to collect predictions and targets for evaluation.
- Use `mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")` to compute mAP.
- Use `cellboxes_to_boxes(output)` and `plot_image(image, bboxes)` to visualize predictions.

You can also run the small model test in `model.py`:
- `python model.py` will execute a small forward pass test (prints output tensor shape).

## Project structure (overview)
- model.py — YOLO v1 model architecture (CNN + detection head)
- train.py — training loop, data loading, hyperparameters
- dataset.py — VOCDataset loader (parses CSV and label files)
- loss.py — YoloLoss implementation
- utils.py — IOU, NMS, mAP, bounding box conversion, plotting, checkpoint helpers
- data/ — expected location for images, labels, and CSVs (not included)

## Dataset & label details (summary)
- CSV: two columns per-row: image_filename, label_filename (no header)
- Label files: whitespace-separated lines, format: class x y w h
  - coordinates normalized to [0, 1]
  - x,y = center of box relative to image
  - w,h = width and height relative to image
- The dataset loader converts each object into a (S, S, C + 5B) label tensor expected by YOLO v1.

## Where to get help
- Open an issue in this repository for bugs, questions, or feature requests: Issues
- Review docstrings and comments in the following files for implementation specifics: `dataset.py`, `utils.py`, `loss.py`, `model.py`, `train.py`.
- If you need a walkthrough for adding a class or visualizing predictions, open a discussion or issue.

## Maintainers & contributing
- Maintainer: noop-rabbit
- Found a bug or want to contribute? Please:
  - Open an issue describing the change or bug
  - Submit a pull request with tests and an explanation
  - See docs/CONTRIBUTING.md or CONTRIBUTING.md if present for contribution guidelines (use relative path in repo)

If CONTRIBUTING.md is not yet present, open an issue first with your proposed change so maintainers can advise.

## License
This project references a LICENSE file in the repository. See `LICENSE` for license details.

---


