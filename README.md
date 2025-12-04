# CIFAR-10 Double Descent Experiments

Final project for GRA4157

## Project Overview

This project tests the double descent phenomenon on CIFAR-10

The CNN architecture is adapted from an MNIST classifier (99% accuracy) to handle RGB images, with parameterized width for controlled experiments.


## Setup

Use UV package manager to create a venv, and use `uv sync` to get all the packages. 

If that does not work, use `uv pip install torch, numpy, matplotlib, ipykernel, seaborn, scikit-learn, torchvision`

## How to run: 

Recommended: Run the `cifar10_experiments_notebook.ipynb` from the top down

You may also run 
```bash 
source .venv/bin/activate

python cifar10_experiments.py
```

Enjoy!