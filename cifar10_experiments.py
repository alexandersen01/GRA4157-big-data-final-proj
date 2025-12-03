import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
from datetime import datetime

from flexible_cnn_classifier import (
    create_baseline_model,
    create_medium_model,
    create_high_model,
    create_very_high_model,
    create_extreme_model,
    create_width_scaled_model,
    get_width_multipliers_for_double_descent,
)
from simple_cnn_classifier import CNNClassifier


def load_cifar10(data_dir="./data", seed=42):  # load cifar dataset with normalization

    # normalization constants
    mean = [
        0.4914,
        0.4822,
        0.4465,
    ]  # thanks dlmacedo https://github.com/kuangliu/pytorch-cifar/issues/19
    std = [0.2470, 0.2435, 0.2616]

    # load training data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # convert to numpy arrays
    X_train_full = []
    y_train_full = []
    for img, label in trainset:
        X_train_full.append(img.numpy())
        y_train_full.append(label)

    X_test = []
    y_test = []
    for img, label in testset:
        X_test.append(img.numpy())
        y_test.append(label)

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # split training into train and validation (70-30 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.3,
        random_state=seed,
        stratify=y_train_full,
    )

    # flatten to (N, 3*32*32) for compatibility with CNNClassifier
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print(f"train set: {X_train.shape[0]} samples")
    print(f"validation set: {X_val.shape[0]} samples")
    print(f"test set: {X_test.shape[0]} samples")
    print(f"classes: {len(np.unique(y_train))}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_names():  # returns cifar class names
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


# MODEL WISE DD


def run_model_wise_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    num_epochs=100,
    learning_rate=0.001,
    results_dir="results/model_wise",
    seed=42,
):
    """
    train models of varying complexity to observe model-wise double descent

    expected: test error shows U-shape then recovery as model complexity increases
    """
    print("model-wise DD experiment")

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # define model configurations
    model_configs = {
        "Baseline": create_baseline_model,
        "Medium": create_medium_model,
        "High": create_high_model,
        "Very High": create_very_high_model,
        "Extreme": create_extreme_model,
    }

    results = {}

    for model_name, model_fn in model_configs.items():
        print(f"Training {model_name} Model")

        start_time = time.time()

        # create model
        model = model_fn(in_channels=3, num_classes=10, seed=seed)
        arch_summary = model.get_architecture_summary()
        print(f"params: {arch_summary['total_parameters']:,}")

        # create classifier
        classifier = CNNClassifier(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            model=model,
            input_shape=(3, 32, 32),
            seed=seed,
        )

        # train with tracking
        history = classifier.fit_with_tracking(
            X_train, y_train, X_val, y_val, batch_size=128, verbose=True
        )

        # evaluate on test set
        y_pred = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        training_time = time.time() - start_time

        print(f"\ntest Set Performance:")
        print(f"  accuracy: {test_accuracy:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  training Time: {training_time:.1f}s")

        # store results
        results[model_name] = {
            "model_name": model_name,
            "architecture": arch_summary,
            "history": history,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_error": 1 - test_accuracy,
            "training_time": training_time,
            "y_pred": y_pred,
        }

        # save checkpoint
        checkpoint_path = (
            Path(results_dir) / f"{model_name.lower().replace(' ', '_')}_checkpoint.pkl"
        )
        with open(checkpoint_path, "wb") as f:
            pickle.dump(results[model_name], f)
        print(f"saved checkpoint to {checkpoint_path}")

        # explicitly free GPU memory before moving on to the next model
        if torch.cuda.is_available():
            del classifier
            del model
            torch.cuda.empty_cache()

    # save consolidated results
    results_path = Path(results_dir) / "all_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved all results to {results_path}")

    return results


# WIDTH-SWEEP DD (like ResNet18 width in the paper)


def run_width_sweep_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    width_multipliers=None,
    num_epochs=200,
    learning_rate=0.001,
    results_dir="results/width_sweep",
    label_noise=0.0,  # Add label noise to make double descent more visible
    train_subset_size=None,  # Use smaller training set for faster interpolation
    seed=42,
):
    """
    Train models of varying WIDTH to observe model-wise double descent.

    This mimics the ResNet18 width experiment from Nakkiran et al. (2019):
    - Many fine-grained width values
    - NO regularization (dropout=0)
    - Optionally add label noise to make the peak more pronounced
    - Optionally use smaller training set to hit interpolation threshold faster

    Args:
        label_noise: Fraction of labels to randomly flip (0.0-0.2 recommended)
        train_subset_size: If set, use only this many training samples
    """
    print("=" * 60)
    print("WIDTH-SWEEP DOUBLE DESCENT EXPERIMENT")
    print("=" * 60)

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if width_multipliers is None:
        width_multipliers = get_width_multipliers_for_double_descent()

    # Optionally subsample training data
    if train_subset_size is not None and train_subset_size < len(X_train):
        np.random.seed(seed)
        indices = np.random.choice(len(X_train), train_subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Using subset of {train_subset_size} training samples")

    # Optionally add label noise
    if label_noise > 0:
        np.random.seed(seed)
        n_noisy = int(len(y_train) * label_noise)
        noisy_indices = np.random.choice(len(y_train), n_noisy, replace=False)
        y_train = y_train.copy()
        y_train[noisy_indices] = np.random.randint(0, 10, n_noisy)
        print(f"Added {label_noise*100:.0f}% label noise ({n_noisy} labels)")

    results = {}

    for width in width_multipliers:
        print(f"\n{'='*40}")
        print(f"Training Width Multiplier = {width}")
        print(f"{'='*40}")

        start_time = time.time()

        # Create width-scaled model (NO dropout!)
        model = create_width_scaled_model(
            width_multiplier=width, in_channels=3, num_classes=10, seed=seed
        )
        arch_summary = model.get_architecture_summary()
        print(f"Parameters: {arch_summary['total_parameters']:,}")

        # Create classifier
        classifier = CNNClassifier(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            model=model,
            input_shape=(3, 32, 32),
            seed=seed,
        )

        # Train with tracking
        history = classifier.fit_with_tracking(
            X_train, y_train, X_val, y_val, batch_size=128, verbose=True
        )

        # Evaluate on test set
        y_pred = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        # Also compute train accuracy (important for seeing interpolation)
        y_train_pred = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        training_time = time.time() - start_time

        print(f"\nResults:")
        print(f"  Train Accuracy: {train_accuracy:.4f} (error: {1-train_accuracy:.4f})")
        print(f"  Test Accuracy: {test_accuracy:.4f} (error: {1-test_accuracy:.4f})")
        print(f"  Training Time: {training_time:.1f}s")

        # Store results
        results[width] = {
            "width_multiplier": width,
            "architecture": arch_summary,
            "history": history,
            "train_accuracy": train_accuracy,
            "train_error": 1 - train_accuracy,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_error": 1 - test_accuracy,
            "training_time": training_time,
            "y_pred": y_pred,
        }

        # Save checkpoint
        checkpoint_path = Path(results_dir) / f"width_{width}_checkpoint.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(results[width], f)

        # Free GPU memory
        if torch.cuda.is_available():
            del classifier
            del model
            torch.cuda.empty_cache()

    # Save consolidated results
    results_path = Path(results_dir) / "all_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved all results to {results_path}")

    return results


def plot_width_sweep_double_descent(
    results, save_path="report/figures/width_sweep_double_descent.pdf"
):
    """Plot test/train error vs model width (like the image you showed)"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    widths = []
    params = []
    train_errors = []
    test_errors = []

    for width, result in sorted(results.items()):
        widths.append(width)
        params.append(result["architecture"]["total_parameters"])
        train_errors.append(result["train_error"])
        test_errors.append(result["test_error"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error vs Width (like your image)
    ax1.plot(
        widths,
        test_errors,
        "o-",
        linewidth=2,
        markersize=6,
        color="steelblue",
        label="Test Error",
    )
    ax1.plot(
        widths,
        train_errors,
        "o--",
        linewidth=2,
        markersize=6,
        color="coral",
        alpha=0.7,
        label="Train Error",
    )
    ax1.fill_between(widths, test_errors, alpha=0.2, color="steelblue")
    ax1.set_xlabel("Model Width Multiplier", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Error", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Double Descent: Error vs Model Width", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Mark interpolation threshold region
    ax1.axvline(
        x=widths[np.argmax(test_errors)],
        color="red",
        linestyle=":",
        alpha=0.5,
        label="Interpolation Threshold",
    )

    # Plot 2: Error vs Parameters (log scale)
    ax2.plot(
        params,
        test_errors,
        "o-",
        linewidth=2,
        markersize=6,
        color="steelblue",
        label="Test Error",
    )
    ax2.plot(
        params,
        train_errors,
        "o--",
        linewidth=2,
        markersize=6,
        color="coral",
        alpha=0.7,
        label="Train Error",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Parameters", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Error", fontsize=12, fontweight="bold")
    ax2.set_title("Double Descent: Error vs Parameters", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()


# EPOCH WISE DD


def run_epoch_wise_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    model_name="Baseline",
    num_epochs=400,
    learning_rate=0.001,
    results_dir="results/epoch_wise",
    checkpoint_path=None,
    resume_from=None,
    seed=42,
):
    """
    train a model for many epochs to observe epoch-wise double descent

    expected: test error improves, degrades, then hopefully recovers
    """
    print("epoch-wise DD")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # default checkpoint path (one rolling checkpoint per model)
    if checkpoint_path is None:
        checkpoint_path = results_dir / f"{model_name.lower()}_epochwise_checkpoint.pkl"
    else:
        checkpoint_path = Path(checkpoint_path)

    # select model
    model_creators = {
        "Baseline": create_baseline_model,
        "Medium": create_medium_model,
        "High": create_high_model,
    }

    # build model
    model = model_creators[model_name](in_channels=3, num_classes=10, seed=seed)

    # load from checkpoint if provided
    start_epoch = 0
    previous_history = None
    if resume_from is not None:
        resume_from = Path(resume_from)
        if resume_from.exists():
            print(f"\nresuming from checkpoint: {resume_from}")
            with open(resume_from, "rb") as f:
                checkpoint = pickle.load(f)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("num_epochs_trained", 0)
            previous_history = checkpoint.get("history", None)
        else:
            print(
                f"\nWARNING: resume checkpoint not found at {resume_from}, starting fresh."
            )

    arch_summary = model.get_architecture_summary()
    print(
        f"\ntraining {model_name} model from epoch {start_epoch} "
        f"for additional {num_epochs} epochs (total target ~{start_epoch + num_epochs})"
    )
    print(f"params: {arch_summary['total_parameters']:,}")

    start_time = time.time()

    # create classifier
    classifier = CNNClassifier(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model=model,
        input_shape=(3, 32, 32),
        seed=seed,
    )

    # train with tracking
    history_segment = classifier.fit_with_tracking(
        X_train, y_train, X_val, y_val, batch_size=128, verbose=True
    )

    # merge history with any previous history from checkpoint
    if previous_history is not None:
        history = {
            key: previous_history.get(key, []) + history_segment.get(key, [])
            for key in history_segment.keys()
        }
        total_epochs_trained = start_epoch + num_epochs
    else:
        history = history_segment
        total_epochs_trained = num_epochs

    # evaluate on test set
    y_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    training_time = time.time() - start_time

    print(f"\nfinal test set performance:")
    print(f"  accuracy: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")

    results = {
        "model_name": model_name,
        "architecture": arch_summary,
        "num_epochs": total_epochs_trained,
        "history": history,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_error": 1 - test_accuracy,
        "training_time": training_time,
        "y_pred": y_pred,
    }

    # save results
    results_path = (
        results_dir / f"{model_name.lower()}_epochs_{total_epochs_trained}_results.pkl"
    )
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved results to {results_path}")

    # save / update rolling checkpoint and clean up older one if needed
    checkpoint_dict = {
        "model_name": model_name,
        "architecture": arch_summary,
        "model_state_dict": model.state_dict(),
        "num_epochs_trained": total_epochs_trained,
        "history": history,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_dict, f)
    print(f"saved checkpoint to {checkpoint_path}")

    return results


# VIZ UTILS
def plot_model_wise_double_descent(
    results, save_path="report/figures/model_wise_double_descent.pdf"
):
    """plot test error vs model complexity"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # extract data
    model_names = []
    params = []
    test_errors = []
    test_f1_scores = []

    for name, result in results.items():
        model_names.append(name)
        params.append(result["architecture"]["total_parameters"])
        test_errors.append(result["test_error"])
        test_f1_scores.append(result["test_f1"])

    # create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # plot 1: test error vs parameters
    ax1.plot(params, test_errors, "o-", linewidth=2, markersize=8, color="steelblue")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Parameters", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Error", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Model-wise Double Descent: Test Error", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # add model labels
    for i, name in enumerate(model_names):
        ax1.annotate(
            name,
            (params[i], test_errors[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # plot 2: F1 Score vs parameters
    ax2.plot(
        params, test_f1_scores, "o-", linewidth=2, markersize=8, color="darkorange"
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Parameters", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Test F1 Score", fontsize=12, fontweight="bold")
    ax2.set_title("Model-wise: Test F1 Score", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # add model labels
    for i, name in enumerate(model_names):
        ax2.annotate(
            name,
            (params[i], test_f1_scores[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_epoch_wise_double_descent(
    results, save_path="report/figures/epoch_wise_double_descent.pdf"
):
    """plot training curves over many epochs"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    history = results["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # plot 1: loss curves
    ax1.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
        linewidth=2,
        color="steelblue",
    )
    ax1.plot(
        epochs, history["val_loss"], label="Val Loss", linewidth=2, color="darkorange"
    )
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # plot 2: val acc
    ax2.plot(epochs, history["val_accuracy"], linewidth=2, color="green")
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("Validation Accuracy Over Time", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # plot 3: val F1
    ax3.plot(epochs, history["val_f1"], linewidth=2, color="purple")
    ax3.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Validation F1 Score", fontsize=12, fontweight="bold")
    ax3.set_title("Validation F1 Score Over Time", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # plot 4: val error (1 - acc)
    val_error = [1 - acc for acc in history["val_accuracy"]]
    ax4.plot(epochs, val_error, linewidth=2, color="red")
    ax4.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Validation Error", fontsize=12, fontweight="bold")
    ax4.set_title(
        "Validation Error (Potential Double Descent)", fontsize=14, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)

    # add horizontal line at minimum error
    min_error = min(val_error)
    ax4.axhline(
        y=min_error,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Min: {min_error:.4f}",
    )
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, class_names, save_path="report/figures/confusion_matrix.pdf"
):
    """plot confusion matrix"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix - CIFAR-10", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def create_results_summary_table(results, save_path="report/figures/results_table.txt"):
    """Create a formatted table of results"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("model-wise DD results summary\n")
        f.write("=" * 100 + "\n\n")

        # header
        f.write(
            f"{'model':<15} {'params':<15} {'test Acc':<12} {'test F1':<12} {'test error':<12} {'time (s)':<10}\n"
        )
        f.write("-" * 100 + "\n")

        # data rows
        for name, result in results.items():
            params = result["architecture"]["total_parameters"]
            acc = result["test_accuracy"]
            f1 = result["test_f1"]
            error = result["test_error"]
            time_s = result["training_time"]

            f.write(
                f"{name:<15} {params:<15,} {acc:<12.4f} {f1:<12.4f} {error:<12.4f} {time_s:<10.1f}\n"
            )

        f.write("=" * 100 + "\n")

    print(f"saved results table to {save_path}")


def main(seed=42):

    MODEL_WISE_NUM_EPOCHS = 0  # Set to 0 to load cached results
    EPOCH_WISE_NUM_EPOCHS = 0  # Set to 0 to load cached results

    # Paths for cached results
    model_wise_cache = Path("results/model_wise/all_results.pkl")
    epoch_wise_cache = Path("results/epoch_wise/baseline_epochs_30000_results.pkl")

    # load data (needed for confusion matrix y_test)
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10(seed=seed)
    class_names = get_class_names()

    # check GPU availability
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # MODEL-WISE EXPERIMENT
    print("model-wise DD")
    if MODEL_WISE_NUM_EPOCHS == 0 and model_wise_cache.exists():
        print(f"Loading cached model-wise results from {model_wise_cache}")
        with open(model_wise_cache, "rb") as f:
            model_wise_results = pickle.load(f)
    else:
        model_wise_results = run_model_wise_experiment(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            num_epochs=MODEL_WISE_NUM_EPOCHS,
            learning_rate=0.001,
            seed=seed,
        )

    # generate model-wise visualizations
    plot_model_wise_double_descent(model_wise_results)
    create_results_summary_table(model_wise_results)

    # plot confusion matrix for best model
    best_model_name = max(
        model_wise_results.keys(), key=lambda k: model_wise_results[k]["test_f1"]
    )
    plot_confusion_matrix(
        y_test,
        model_wise_results[best_model_name]["y_pred"],
        class_names,
        save_path=f'report/figures/confusion_matrix_{best_model_name.lower().replace(" ", "_")}.pdf',
    )

    # EPOCH-WISE EXPERIMENT
    print("epoch-wise DD")
    if EPOCH_WISE_NUM_EPOCHS == 0 and epoch_wise_cache.exists():
        print(f"Loading cached epoch-wise results from {epoch_wise_cache}")
        with open(epoch_wise_cache, "rb") as f:
            epoch_wise_results = pickle.load(f)
    else:
        epoch_wise_results = run_epoch_wise_experiment(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            model_name="Baseline",
            num_epochs=EPOCH_WISE_NUM_EPOCHS,
            learning_rate=0.001,
            seed=seed,
        )

    # generate epoch-wise vis
    plot_epoch_wise_double_descent(epoch_wise_results)


if __name__ == "__main__":
    main()
