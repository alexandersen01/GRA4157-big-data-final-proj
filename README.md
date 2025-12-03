# CIFAR-10 Double Descent Experiments

Final project for GRA4157 - Big Data, investigating Belkin et al.'s double descent theory on CIFAR-10.

## Project Overview

This project tests the double descent phenomenon on CIFAR-10 through two complementary experiments:

1. **Width-Sweep Double Descent**: Training models of varying width (fine-grained model-wise DD)
2. **Epoch-wise Double Descent**: Training a larger model for many epochs

The CNN architecture is adapted from an MNIST classifier (99% accuracy) to handle RGB images, with parameterized width for controlled experiments.

## Repository Structure

```
.
├── cifar10_experiments_notebook.ipynb  # Jupyter notebook (RECOMMENDED)
├── flexible_cnn_classifier.py          # Parameterized CNN with width scaling
├── simple_cnn_classifier.py            # Training wrapper with tracking
├── simple_bootleg_cnn_classifier.py    # Original MNIST baseline (for reference)
├── cifar10_experiments.py              # Command-line experiment runner
├── report/
│   ├── report.tex                      # Main report
│   ├── sources.bib                     # Bibliography (Belkin citation)
│   └── figures/                        # Generated plots
├── results/                            # Experiment checkpoints and results
│   ├── width_sweep/                    # Width-sweep DD results
│   └── epoch_wise/                     # Epoch-wise DD results
└── data/                               # CIFAR-10 dataset cache
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but works on CPU/MPS)
- Virtual environment (recommended)
- [`uv`](https://github.com/astral-sh/uv) for dependency and environment management

### Installation

```bash
# Install dependencies and create .venv from pyproject.toml / uv.lock
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Running Experiments

### Option 1: Using Jupyter Notebook (Recommended)

The easiest way to run experiments with full control:

```bash
source .venv/bin/activate  # ensure the uv-created venv is active
jupyter notebook cifar10_experiments_notebook.ipynb
```

The notebook includes:

- Single `SEED` constant for reproducibility
- Interactive progress monitoring
- Inline visualizations
- Easy configuration of experiment parameters
- Step-by-step execution

### Option 2: Command Line

Test the architecture on a small batch:

```bash
source .venv/bin/activate
python flexible_cnn_classifier.py
```

Run both experiments from command line:

```bash
source .venv/bin/activate
python cifar10_experiments.py
```

This will:

- Download CIFAR-10 (if not cached)
- Train width-sweep models (width 1-128) for model-wise DD experiment
- Train 1 larger model for many epochs for epoch-wise DD experiment
- Generate all figures in `report/figures/`
- Save results to `results/`

### Run Individual Experiments

You can modify `cifar10_experiments.py` to run only one experiment by commenting out the other in the `main()` function.

## Model Architecture

The width-sweep experiment uses a 2-block CNN with parameterized width:

| Width Multiplier | Base Filters | FC Hidden | Approx. Parameters |
| ---------------- | ------------ | --------- | ------------------ |
| 1                | 8            | 32        | ~3,000             |
| 10               | 80           | 320       | ~300,000           |
| 20               | 160          | 640       | ~1,200,000         |
| 64               | 512          | 2048      | ~12,000,000        |
| 128              | 1024         | 4096      | ~50,000,000        |

Key settings for observing double descent:
- **No dropout** (dropout=0) to avoid smoothing out the interpolation peak
- **Label noise** (15%) to make the peak more pronounced
- **Smaller training subset** (10k samples) to hit interpolation threshold faster

## Expected Results

### Width-Sweep Double Descent (Model-wise)

According to Belkin/Nakkiran's theory, we expect test error to:

1. Decrease initially (underparameterized regime, small widths)
2. **Peak** near the interpolation threshold (where model can just fit training data)
3. Decrease again (overparameterized regime, large widths)

### Epoch-wise Double Descent

With a model that can interpolate training data, we expect:

1. Improve during initial training
2. Degrade as model overfits (approaches interpolation)
3. Potentially recover with continued training past interpolation

**Note**: CIFAR-10 may not exhibit strong double descent. Negative results are still scientifically valuable!

## Hardware Requirements

- **Minimum**: CPU, 8GB RAM, ~50GB disk space
  - Training time: ~24-48 hours for all experiments
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
  - Training time: ~4-8 hours for all experiments
- **Used in project**: NVIDIA RTX 5090 via SSH
  - Training time: ~2-4 hours for all experiments

## Results

After running experiments, results will be saved to:

- `results/width_sweep/all_results.pkl` - Width-sweep DD results
- `results/epoch_wise/*.pkl` - Epoch-wise DD results
- `report/figures/*.pdf` - Publication-ready figures

## Compiling the Report

```bash
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Or use your preferred LaTeX IDE (Overleaf, TeXShop, etc.)

## Key Dependencies

- PyTorch (with CUDA support recommended)
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy

See `pyproject.toml` / `uv.lock` for the complete, version-locked dependency set.

## References

- Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. _PNAS_, 116(32), 15849-15854.
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## Author

Jakob Sverre Alexandersen  
Email: ja.al@fsncapital.com  
Course: GRA4157 - Big Data

## License

Academic project - for educational purposes only.
