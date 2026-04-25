import csv
import random
import re

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def save_training_curve(history, csv_path, plot_path=None, title="Training Curve"):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)

    if plot_path is None or plt is None:
        return None

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_accs = [row["val_accuracy"] for row in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, color="tab:blue", marker="o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, color="tab:orange", marker="s", label="Val Accuracy")
    ax2.set_ylabel("Val Accuracy (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path