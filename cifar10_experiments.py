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
import gc

from flexible_cnn_classifier import (
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


# WIDTH-SWEEP DD (like ResNet18 width in the paper)


def clear_gpu_memory():
    """Aggressively clear GPU memory to prevent OOM between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        # MPS doesn't have empty_cache, but gc.collect helps
        pass


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
    resume=True,  # Resume from existing checkpoints
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
        resume: If True, skip widths that already have checkpoints with same num_epochs
    """
    print("=" * 60)
    print("WIDTH-SWEEP DOUBLE DESCENT EXPERIMENT")
    print(f"Training for {num_epochs} epochs per model")
    print("=" * 60)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    total_widths = len(width_multipliers)

    for idx, width in enumerate(width_multipliers):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total_widths}] Training Width Multiplier = {width}")
        print(f"{'='*60}")

        checkpoint_path = results_dir / f"width_{width}_checkpoint.pkl"

        # Check if we can resume from existing checkpoint
        if resume and checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                existing = pickle.load(f)
            existing_epochs = len(existing.get("history", {}).get("train_loss", []))
            if existing_epochs >= num_epochs:
                print(
                    f"  Skipping - already trained for {existing_epochs} epochs (>= {num_epochs})"
                )
                results[width] = existing
                continue
            else:
                print(
                    f"  Found checkpoint with {existing_epochs} epochs, retraining for {num_epochs}..."
                )

        # Clear GPU memory before creating new model
        clear_gpu_memory()

        start_time = time.time()

        # Create width-scaled model (NO dropout!)
        model = create_width_scaled_model(
            width_multiplier=width, in_channels=3, num_classes=10, seed=seed
        )
        arch_summary = model.get_architecture_summary()
        print(f"  Parameters: {arch_summary['total_parameters']:,}")

        # Estimate time
        estimated_time_per_epoch = 0.18 + (width * 0.01)  # rough estimate
        estimated_total = estimated_time_per_epoch * num_epochs / 60
        print(f"  Estimated time: ~{estimated_total:.0f} minutes")

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

        print(f"\n  Results:")
        print(
            f"    Train Accuracy: {train_accuracy:.4f} (error: {1-train_accuracy:.4f})"
        )
        print(f"    Test Accuracy: {test_accuracy:.4f} (error: {1-test_accuracy:.4f})")
        print(f"    Training Time: {training_time:.1f}s ({training_time/60:.1f} min)")

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
            "num_epochs": num_epochs,
            "y_pred": y_pred,
        }

        # Save checkpoint immediately after each model
        with open(checkpoint_path, "wb") as f:
            pickle.dump(results[width], f)
        print(f"  Saved checkpoint to {checkpoint_path}")

        # Aggressively free GPU memory
        del classifier.model
        del classifier
        del model
        del y_pred
        del y_train_pred
        clear_gpu_memory()

    # Save consolidated results
    results_path = results_dir / "all_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved all results to {results_path}")

    return results


def plot_width_sweep_double_descent(
    results, save_path="report/figures/width_sweep_double_descent.pdf"
):
    """Plot test/train error vs model width (Nakkiran et al. style)"""
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

    # Set dark style like the reference image
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot test error with confidence band effect
    ax.plot(
        widths,
        test_errors,
        "-",
        linewidth=2.5,
        color="#6A5ACD",  # Slate blue like the reference
        label="Reality",
        zorder=3,
    )
    ax.fill_between(
        widths,
        [e - 0.01 for e in test_errors],
        [e + 0.01 for e in test_errors],
        alpha=0.3,
        color="#6A5ACD",
        zorder=2,
    )

    # Find the peak (interpolation threshold)
    peak_idx = np.argmax(test_errors)
    peak_width = widths[peak_idx]
    peak_error = test_errors[peak_idx]

    # Add "Expected" annotation lines (like the reference)
    # Modern ML expectation (monotonic decrease)
    ax.annotate(
        "Expected\n(Modern ML)",
        xy=(2, test_errors[1] + 0.02),
        xytext=(3, 0.65),
        fontsize=10,
        color="#FFA500",  # Orange
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#FFA500", lw=1.5),
    )

    # Classical statistics expectation (U-shape, keeps going up)
    ax.annotate(
        "Expected\n(Classical Statistics)",
        xy=(peak_width, peak_error + 0.02),
        xytext=(15, 0.65),
        fontsize=10,
        color="#32CD32",  # Lime green
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#32CD32", lw=1.5),
    )

    # Add "Reality" label near the curve
    mid_idx = len(widths) // 3
    ax.annotate(
        "Reality",
        xy=(widths[mid_idx], test_errors[mid_idx]),
        xytext=(widths[mid_idx] - 5, test_errors[mid_idx] - 0.05),
        fontsize=11,
        color="#6A5ACD",
        fontweight="bold",
    )

    # Axis styling
    ax.set_xlabel("Model Size (Width Multiplier)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test / Train Error", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(widths) + 2)
    ax.set_ylim(0.2, 0.75)

    # Grid
    ax.grid(True, alpha=0.2, linestyle="-")
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="#1a1a2e")
    print(f"Saved figure to {save_path}")
    plt.close()

    # Reset style for other plots
    plt.style.use("default")

    # Also save a second version with both train and test error
    plt.style.use("dark_background")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(
        widths, test_errors, "-", linewidth=2.5, color="#6A5ACD", label="Test Error"
    )
    ax2.plot(
        widths,
        train_errors,
        "--",
        linewidth=2,
        color="#FF6B6B",
        alpha=0.8,
        label="Train Error",
    )
    ax2.fill_between(widths, test_errors, alpha=0.2, color="#6A5ACD")

    ax2.axvline(
        x=peak_width,
        color="#FFD700",
        linestyle=":",
        alpha=0.5,
        label=f"Interpolation Threshold (k={peak_width})",
    )

    ax2.set_xlabel("Model Size (Width Multiplier)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Error", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Double Descent: Width Sweep Experiment",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, max(widths) + 2)

    plt.tight_layout()
    save_path2 = save_path.replace(".pdf", "_detailed.pdf")
    plt.savefig(save_path2, bbox_inches="tight", dpi=300, facecolor="#1a1a2e")
    print(f"Saved detailed figure to {save_path2}")
    plt.close()

    plt.style.use("default")


# EPOCH WISE DD


def run_epoch_wise_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    width_multiplier=20,
    num_epochs=400,
    learning_rate=0.001,
    results_dir="results/epoch_wise",
    checkpoint_path=None,
    resume_from=None,
    seed=42,
):
    """
    Train a model for many epochs to observe epoch-wise double descent.

    Uses width-scaled model (no dropout) that can interpolate training data.
    Expected: test error improves, degrades at interpolation, then recovers.

    Args:
        width_multiplier: Width scaling factor for model (higher = more params).
                         Use ~20 for a model that can interpolate ~10k samples.
    """
    print("epoch-wise DD")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"width_{width_multiplier}"

    # default checkpoint path (one rolling checkpoint per model)
    if checkpoint_path is None:
        checkpoint_path = results_dir / f"{model_name}_epochwise_checkpoint.pkl"
    else:
        checkpoint_path = Path(checkpoint_path)

    # build width-scaled model (no dropout for epoch-wise DD)
    model = create_width_scaled_model(
        width_multiplier=width_multiplier, in_channels=3, num_classes=10, seed=seed
    )

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
    """Create a formatted table of results for width-sweep experiment"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write("Width-Sweep Double Descent Results Summary\n")
        f.write("=" * 110 + "\n\n")

        # header
        f.write(
            f"{'Width':<10} {'Params':<15} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Test Error':<12} {'Time (s)':<10}\n"
        )
        f.write("-" * 110 + "\n")

        # data rows (sorted by width)
        for width, result in sorted(results.items()):
            params = result["architecture"]["total_parameters"]
            train_acc = result.get("train_accuracy", 0)
            acc = result["test_accuracy"]
            f1 = result["test_f1"]
            error = result["test_error"]
            time_s = result["training_time"]

            f.write(
                f"{width:<10} {params:<15,} {train_acc:<12.4f} {acc:<12.4f} {f1:<12.4f} {error:<12.4f} {time_s:<10.1f}\n"
            )

        f.write("=" * 110 + "\n")

    print(f"saved results table to {save_path}")


def main(seed=42):

    WIDTH_SWEEP_NUM_EPOCHS = 4000  # 4000 epochs for double descent (like the paper)
    EPOCH_WISE_NUM_EPOCHS = 0  # Set to 0 to load cached results

    # Paths for cached results - use new directory for 4000 epoch run
    results_dir = "results/width_sweep_4000"
    width_sweep_cache = Path(f"{results_dir}/all_results.pkl")
    epoch_wise_cache = Path("results/epoch_wise/baseline_epochs_30000_results.pkl")

    # load data
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
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # WIDTH-SWEEP EXPERIMENT (Model-wise Double Descent)
    print("\n" + "=" * 60)
    print("WIDTH-SWEEP DOUBLE DESCENT (4000 EPOCHS)")
    print("=" * 60)
    if WIDTH_SWEEP_NUM_EPOCHS == 0 and width_sweep_cache.exists():
        print(f"Loading cached width-sweep results from {width_sweep_cache}")
        with open(width_sweep_cache, "rb") as f:
            width_sweep_results = pickle.load(f)
    else:
        width_sweep_results = run_width_sweep_experiment(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            num_epochs=WIDTH_SWEEP_NUM_EPOCHS,
            results_dir=results_dir,
            label_noise=0.15,  # Add label noise to make double descent peak more visible
            train_subset_size=10000,  # Smaller subset to hit interpolation faster
            learning_rate=0.001,
            seed=seed,
            resume=True,  # Resume from checkpoints if interrupted
        )

    # generate width-sweep vis
    plot_width_sweep_double_descent(
        width_sweep_results,
        save_path="report/figures/width_sweep_double_descent_4000.pdf",
    )
    create_results_summary_table(
        width_sweep_results, save_path="report/figures/width_sweep_results_4000.txt"
    )

    # EPOCH-WISE EXPERIMENT
    print("\n" + "=" * 60)
    print("EPOCH-WISE DOUBLE DESCENT")
    print("=" * 60)
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
            width_multiplier=20,  # Larger model that can interpolate training data
            num_epochs=EPOCH_WISE_NUM_EPOCHS,
            learning_rate=0.001,
            seed=seed,
        )

    # generate epoch-wise vis
    plot_epoch_wise_double_descent(epoch_wise_results)


if __name__ == "__main__":
    main()
