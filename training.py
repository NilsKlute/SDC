# train.py
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from network import ClassificationNetwork
from data import DrivingDatasetHWC

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ---------------- CONFIGURATION -------------- #

VERBOSE: bool = True
PRINT_EVERY: int = 50
VAL_PRINT_EVERY: int = 10
FINETUNE: bool = True

OLD_MODEL_PATH: str = (
    "/media/sn/Frieder_Data/Master_Machine_Learning/Self-Driving-Cars/SDC/models/"
    "hyperconfig_dataset:678868_bs:256_conv_n:4_lin_n:3_aug=True_drop=0.2_epochs:75_lr:0.001_gamma:0.5/agent.pth"
)

# Hyperparameter sets
NR_CONV_LAYERS: int = 3
NR_LINEAR_LAYERS: int = 2
DATA_AUGMENTATION: bool = True
USE_DROPOUT: bool = True

DROPOUT_PARAMS = [0.2] if USE_DROPOUT else [0.0]
LR_PARAMS = [1e-4]
BATCHSIZE_PARAMS = [128]
GAMMA_PARAMS = [0.5]
DATASET_PROP_PARAMS = [1.0]


# ---------------- Utils -------------------- #

def print_info(msg: str) -> None:
    """Print message if verbose mode is on."""
    if VERBOSE:
        print(msg, flush=True)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate of the optimizer."""
    return optimizer.param_groups[0].get("lr", 0.0)


# ---------------- MAIN TRAIN FUNCTION ---------------- #

def train(data_folder: str, trained_network_file: str, args: Any) -> None:
    """
    Train the classification model on the dataset.

    Args:
        data_folder (str): Path containing 'observations.npy' and 'actions.npy'.
        trained_network_file (str): Filename for saving model weights.
        args (Any): Object with `nr_epochs` attribute.
    """
    print_info("=== Starting training ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")

    # GPU info
    if device.type == "cuda":
        print_info(f"CUDA devices available: {torch.cuda.device_count()}")
        print_info(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print_info("CUDA not available — using CPU.")

    # Paths
    obs_path = os.path.join(data_folder, "observations.npy")
    act_path = os.path.join(data_folder, "actions.npy")
    print_info(f"Observations: {obs_path}")
    print_info(f"Actions: {act_path}")

    # ---- Load Data ----
    actions_np = np.load(act_path, mmap_mode="r")
    print_info(f"Loaded actions with shape={actions_np.shape}")

    helper = ClassificationNetwork()

    def to_class_fn(action_tensor: torch.Tensor) -> int:
        """Convert action tensor to class index."""
        classes, _ = helper.actions_to_classes([action_tensor])
        c0 = classes[0]
        return int(c0.item() if torch.is_tensor(c0) else c0)

    full_dataset = DrivingDatasetHWC(
        obs_path=obs_path,
        actions_array=actions_np,
        to_class_fn=to_class_fn,
        train=True,
        augment=DATA_AUGMENTATION,
    )

    total_samples = len(full_dataset)
    print_info(f"Total dataset size: {total_samples}")

    # ---- Split ----
    split_size = int(0.9 * total_samples)
    train_ds, val_ds = random_split(full_dataset, [split_size, total_samples - split_size])
    val_ds.dataset.train = False
    val_ds.dataset.augment = False
    print_info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- DataLoader Factory ----
    def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
        num_workers = min(8, os.cpu_count() or 2)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(os.cpu_count() or 0) > 1,
        )

    # ---------------- HYPERPARAMETER LOOP ---------------- #

    for dropout in DROPOUT_PARAMS:
        for lr in LR_PARAMS:
            for batch_size in BATCHSIZE_PARAMS:
                for dataset_prop in DATASET_PROP_PARAMS:
                    for gamma in GAMMA_PARAMS:

                        sub_len = int(dataset_prop * len(train_ds))
                        train_subset = Subset(train_ds, range(sub_len))
                        print_info(f"Using {sub_len} samples for training (subset of dataset).")

                        train_loader = make_loader(train_subset, batch_size, shuffle=True)
                        val_loader = make_loader(val_ds, batch_size, shuffle=False)

                        # ---- Model, Optimizer, Scheduler ---- #
                        model = ClassificationNetwork(dropout).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=gamma)
                        scaler = GradScaler(enabled=True)
                        loss_fn = nn.CrossEntropyLoss()

                        # ---- Fine-tuning ---- #
                        if FINETUNE:
                            state = torch.load(OLD_MODEL_PATH, map_location=device)
                            state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
                            model.load_state_dict(state_dict, strict=True)
                            print_info(f"Loaded pretrained weights from {OLD_MODEL_PATH}")

                        # ---- Save Paths ---- #
                        dataset_size = len(train_subset) + len(val_ds)
                        model_dir = (
                            f"models/hyperconfig_dataset:{dataset_size}"
                            f"_bs:{batch_size}_conv_n:{NR_CONV_LAYERS}"
                            f"_lin_n:{NR_LINEAR_LAYERS}_aug={DATA_AUGMENTATION}"
                            f"_drop={dropout}_epochs:{args.nr_epochs}_lr:{lr}_gamma:{gamma}"
                        )
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, trained_network_file)
                        print_info(f"Saving model checkpoints to: {model_path}")

                        # ---- Training Setup ---- #
                        best_val_loss = float("inf")
                        best_epoch = 0
                        patience = 10
                        no_improve = 0
                        history = []

                        # ------------ EPOCH LOOP ------------ #

                        for epoch in range(args.nr_epochs):
                            if no_improve >= patience:
                                print_info(f"Early stopping (no improvement in {patience} epochs).")
                                break

                            print_info(f"\nEpoch {epoch + 1}/{args.nr_epochs} (lr={current_lr(optimizer):.2e})")

                            # ---- TRAIN ---- #
                            model.train()
                            train_loss_sum = 0.0

                            for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                                xb, yb = xb.to(device), yb.to(device)
                                optimizer.zero_grad(set_to_none=True)

                                with autocast():
                                    logits = model(xb)
                                    loss = loss_fn(logits, yb)

                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                                train_loss_sum += float(loss.item())

                                if batch_idx % PRINT_EVERY == 0:
                                    print_info(f"[Train] Batch {batch_idx}: loss={loss.item():.6f}")

                            scheduler.step()
                            avg_train_loss = train_loss_sum / max(1, len(train_loader))
                            print_info(f"Train loss: {avg_train_loss:.6f}")

                            # ---- VALIDATE ---- #
                            model.eval()
                            val_loss_sum = 0.0

                            with torch.no_grad():
                                for batch_idx, (xb, yb) in enumerate(val_loader, start=1):
                                    xb, yb = xb.to(device), yb.to(device)
                                    with autocast():
                                        logits = model(xb)
                                        loss = loss_fn(logits, yb)
                                    val_loss_sum += float(loss.item())

                                    if batch_idx % VAL_PRINT_EVERY == 0:
                                        print_info(f"[Val] Batch {batch_idx}: loss={loss.item():.6f}")

                            avg_val_loss = val_loss_sum / max(1, len(val_loader))
                            print_info(f"Validation loss: {avg_val_loss:.6f}")
                            history.append([avg_train_loss, avg_val_loss])

                            # ---- Early Stopping ----
                            if avg_val_loss < best_val_loss - 1e-8:
                                best_val_loss = avg_val_loss
                                best_epoch = epoch
                                no_improve = 0
                                torch.save(model.state_dict(), model_path)
                                print_info(f"Saved best model (val_loss={best_val_loss:.6f})")
                            else:
                                no_improve += 1
                                print_info(f"No improvement ({no_improve}/{patience})")

                        print_info(f"Training done. Best epoch={best_epoch + 1}, val_loss={best_val_loss:.6f}")

                        # ---- Plot Loss Curves ----
                        train_losses, val_losses = zip(*history)
                        plt.figure()
                        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
                        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
                        plt.xlabel("Epochs")
                        plt.ylabel("Loss")
                        plt.title("Training and Validation Loss")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(model_dir, "loss_plot.png"))
                        plt.close()

                        np.save(os.path.join(model_dir, "train_val_loss.npy"), np.array(history, dtype=np.float32))
                        print_info(f"Saved results to {model_dir}")

    print_info("=== Training complete ===")