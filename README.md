# CIFAR-10 Double Descent Experiments

Final project for GRA4157 - Big Data, investigating Belkin et al.'s double descent theory on CIFAR-10.

## Project Overview

This project tests the double descent phenomenon on CIFAR-10 through two complementary experiments:

1. **Model-wise Double Descent**: Training models of varying complexity (105k to 8M parameters)
2. **Epoch-wise Double Descent**: Training a single model for 400+ epochs

The baseline CNN architecture is adapted from an MNIST classifier (99% accuracy) to handle RGB images.

## Repository Structure

```
.
├── cifar10_experiments_notebook.ipynb  # Jupyter notebook (RECOMMENDED)
├── flexible_cnn_classifier.py          # Parameterized CNN with seed support
├── simple_cnn_classifier.py            # Wrapper with tracking and seed support
├── simple_bootleg_cnn_classifier.py    # Original MNIST baseline
├── cifar10_experiments.py              # Command-line experiment runner
├── test_setup.py                       # Setup verification script
├── quick_test_experiment.py            # Quick validation runner
├── requirements.txt                    # Python dependencies
├── report/
│   ├── report.tex                      # Main report with Methods section
│   ├── sources.bib                     # Bibliography (Belkin citation)
│   └── figures/                        # Generated plots
├── results/                            # Experiment checkpoints and results
│   ├── model_wise/
│   └── epoch_wise/
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
- Train 4 models for model-wise experiment (~2-4 hours on RTX 5090)
- Train 1 model for 400 epochs for epoch-wise experiment (~2-3 hours)
- Generate all figures in `report/figures/`
- Save results to `results/`

### Run Individual Experiments

You can modify `cifar10_experiments.py` to run only one experiment by commenting out the other in the `main()` function.

## Model Architectures

| Model     | Conv Blocks | Base Filters | FC Hidden | Parameters |
| --------- | ----------- | ------------ | --------- | ---------- |
| Baseline  | 2           | 16           | 64        | ~105,000   |
| Medium    | 3           | 32           | 128       | ~500,000   |
| High      | 4           | 64           | 256       | ~2,000,000 |
| Very High | 4           | 128          | 512       | ~8,000,000 |

## Expected Results

### Model-wise Double Descent

According to Belkin's theory, we expect test error to:

1. Decrease initially (underparameterized regime)
2. Potentially increase near interpolation threshold
3. Decrease again (overparameterized regime)

### Epoch-wise Double Descent

We expect test error to:

1. Improve during initial training
2. Plateau or slightly degrade (overfitting)
3. Potentially recover with continued training

**Note**: CIFAR-10 may not exhibit strong double descent due to dataset characteristics. Negative results are still scientifically valuable!

## Hardware Requirements

- **Minimum**: CPU, 8GB RAM, ~50GB disk space
  - Training time: ~24-48 hours for all experiments
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
  - Training time: ~4-8 hours for all experiments
- **Used in project**: NVIDIA RTX 5090 via SSH
  - Training time: ~2-4 hours for all experiments

## Results

After running experiments, results will be saved to:

- `results/model_wise/all_results.pkl` - Model-wise results
- `results/epoch_wise/*.pkl` - Epoch-wise results
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
