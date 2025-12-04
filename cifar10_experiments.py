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
    label_noise=0.0,  # add label noise to make double descent more visible
    train_subset_size=None,  # use smaller training set for faster interpolation
    seed=42,
    resume=True,  # resume from existing checkpoints if code crashes
):
    """
    Train models of varying WIDTH to observe model-wise double descent.

    Args:
        label_noise: Fraction of labels to randomly flip (0.0-0.2 recommended)
        train_subset_size: If set, use only this many training samples
        resume: If True, skip widths that already have checkpoints with same num_epochs
    """
    print("=" * 60)
    print("WIDTH-SWEEP DOUBLE DESCENT EXPERIMENT")
    print(f"training for {num_epochs} epochs per model")
    print("=" * 60)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if width_multipliers is None:
        width_multipliers = get_width_multipliers_for_double_descent()

    # optionally subsample training data
    if train_subset_size is not None and train_subset_size < len(X_train):
        np.random.seed(seed)
        indices = np.random.choice(len(X_train), train_subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"using subset of {train_subset_size} training samples")

    # optionally add label noise
    if label_noise > 0:
        np.random.seed(seed)
        n_noisy = int(len(y_train) * label_noise)
        noisy_indices = np.random.choice(len(y_train), n_noisy, replace=False)
        y_train = y_train.copy()
        y_train[noisy_indices] = np.random.randint(0, 10, n_noisy)
        print(f"added {label_noise*100:.0f}% label noise ({n_noisy} labels)")

    results = {}
    total_widths = len(width_multipliers)

    for idx, width in enumerate(width_multipliers):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total_widths}] training width multiplier = {width}")
        print(f"{'='*60}")

        checkpoint_path = results_dir / f"width_{width}_checkpoint.pkl"

        # check if we can resume from existing checkpoint
        if resume and checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                existing = pickle.load(f)
            existing_epochs = len(existing.get("history", {}).get("train_loss", []))
            if existing_epochs >= num_epochs:
                print(
                    f"  skipping - already trained for {existing_epochs} epochs (>= {num_epochs})"
                )
                results[width] = existing
                continue
            else:
                print(
                    f"  found checkpoint with {existing_epochs} epochs, retraining for {num_epochs}..."
                )

        # clear GPU memory before creating new model
        clear_gpu_memory()

        start_time = time.time()

        # create width-scaled model (NO dropout!)
        model = create_width_scaled_model(
            width_multiplier=width, in_channels=3, num_classes=10, seed=seed
        )
        arch_summary = model.get_architecture_summary()
        print(f"  Parameters: {arch_summary['total_parameters']:,}")

        # estimate time
        estimated_time_per_epoch = 0.18 + (width * 0.01)  # rough estimate
        estimated_total = estimated_time_per_epoch * num_epochs / 60
        print(f"  Estimated time: ~{estimated_total:.0f} minutes")

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

        # also compute train accuracy (important for seeing interpolation)
        y_train_pred = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        training_time = time.time() - start_time

        print(f"\n  rresults:")
        print(
            f"    train accuracy: {train_accuracy:.4f} (error: {1-train_accuracy:.4f})"
        )
        print(f"    test accuracy: {test_accuracy:.4f} (error: {1-test_accuracy:.4f})")
        print(f"    training time: {training_time:.1f}s ({training_time/60:.1f} min)")

        # store results
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

        # save checkpoint immediately after each model
        with open(checkpoint_path, "wb") as f:
            pickle.dump(results[width], f)
        print(f"  Saved checkpoint to {checkpoint_path}")

        # aggressively free GPU memory
        del classifier.model
        del classifier
        del model
        del y_pred
        del y_train_pred
        clear_gpu_memory()

    # save consolidated results
    results_path = results_dir / "all_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved all results to {results_path}")

    return results


def create_results_summary_table(results, save_path="report/figures/results_table.txt"):
    """Create a formatted table of results for width-sweep experiment"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write("width-sweep DD results summary\n")
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

    WIDTH_SWEEP_NUM_EPOCHS = 200

    # paths for cached results
    results_dir = "results/width_sweep_200"
    width_sweep_cache = Path(f"{results_dir}/all_results.pkl")

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10(seed=seed)

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
    print("WIDTH-SWEEP DOUBLE DESCENT (200 EPOCHS)")
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
            label_noise=0.15,  # add label noise to make double descent peak more visible
            train_subset_size=10000,  # smaller subset to hit interpolation faster
            learning_rate=0.001,
            seed=seed,
            resume=True,  # resume from checkpoints if interrupted
        )

    create_results_summary_table(
        width_sweep_results, save_path="report/figures/width_sweep_results_200.txt"
    )


if __name__ == "__main__":
    main()
