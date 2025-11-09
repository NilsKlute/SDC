# train.py
import os, time, random, contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from network import ClassificationNetwork
from data import DrivingDatasetHWC

# ---- Verbosity controls ----
VERBOSE = True
PRINT_EVERY = 1       # print every N training batches (set 10/50 if too spammy)
VAL_PRINT_EVERY = 10  # print every N validation batches

# Keep TF32 disabled (matches your original)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# HYPERPARAMETERS
nr_conv_layers = 3
nr_linear_layers = 2
data_augmentation = False
use_dropout = True

dropout_params = [0.2] if use_dropout else [0.0]
lr_params = [1e-3]
batchsize_params = [256]
gamma_params = [0.5]
dataset_prop_params = [1]

def _ts():
    return time.strftime("%H:%M:%S")

def log(msg):
    if VERBOSE:
        print(f"[{_ts()}] {msg}", flush=True)

@contextlib.contextmanager
def time_block(name):
    t0 = time.perf_counter()
    log(f"--> {name} START")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"<-- {name} END ({dt:.3f}s)")

def current_lr(optim):
    return optim.param_groups[0].get("lr", None)

def train(data_folder, trained_network_file, args):
    log("Starting train()")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"CUDA devices: {torch.cuda.device_count()}")
        log(f"Current device: {torch.cuda.current_device()}")
        log(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        log("CUDA not available; running on CPU")

    obs_path = os.path.join(data_folder, 'observations.npy')   # (N,96,96,3)
    act_path = os.path.join(data_folder, 'actions.npy')        # (N,A)
    log(f"Observations path: {obs_path}")
    log(f"Actions path: {act_path}")

    with time_block("Load actions (memmap)"):
        actions_np = np.load(act_path, mmap_mode="r")
        log(f"actions_np shape={actions_np.shape}, dtype={actions_np.dtype}, writable={actions_np.flags.writeable}")

    # Helper for class mapping (same as before)
    helper = ClassificationNetwork()

    def to_class_fn(action_tensor):
        # returns index; helper.actions_to_classes returns (classes, weights)
        classes = helper.actions_to_classes([action_tensor])
        c0 = classes[0]
        return int(c0.item() if torch.is_tensor(c0) else c0)

    with time_block("Create dataset (HWC, 0..255)"):
        full_ds = DrivingDatasetHWC(
            obs_path=obs_path,
            actions_array=actions_np,
            to_class_fn=to_class_fn,
            train=True,
            augment=data_augmentation,
        )
        N = len(full_ds)
        log(f"Full dataset size N={N}")

    with time_block("Train/Val split 90/10"):
        split_n = int(0.9 * N)
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [split_n, N - split_n], generator=gen)
        val_ds.dataset.train = False
        val_ds.dataset.augment = False
        log(f"Train size={len(train_ds)}  Val size={len(val_ds)}  Augment(train)={data_augmentation}")

    def make_loader(ds, bs, shuffle):
        nw = min(8, os.cpu_count() or 2)
        pin = True
        persist = (os.cpu_count() or 0) > 1
        log(f"DataLoader: bs={bs}, shuffle={shuffle}, num_workers={nw}, pin_memory={pin}, persistent_workers={persist}")
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=persist,
        )

    for dropout in dropout_params:
        for lr in lr_params:
            for batch_size in batchsize_params:
                for dataset_prop in dataset_prop_params:
                    for gamma in gamma_params:

                        # Subsample train by proportion
                        sub_len = int(dataset_prop * len(train_ds))
                        train_sub = Subset(train_ds, range(sub_len))
                        log(f"Dataset proportion={dataset_prop} -> train_sub={sub_len}")

                        log("Making train/val loaders…")
                        with time_block("Create DataLoaders"):
                            train_loader = make_loader(train_sub, batch_size, shuffle=True)
                            val_loader = make_loader(val_ds, batch_size, shuffle=False)

                        # Model + optimizer
                        log("Building model/optimizer/scheduler/scaler")
                        model = ClassificationNetwork(dropout).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=gamma)
                        scaler = GradScaler(enabled=True)

                        # Loss (unweighted here, matches your code)
                        loss_fn = nn.CrossEntropyLoss()

                        # Folder setup
                        dataset_size = len(train_sub) + len(val_ds)
                        model_folder = (
                            f"models/hyperconfig_dataset:{dataset_size}"
                            f"_bs:{batch_size}_conv_n:{nr_conv_layers}"
                            f"_lin_n:{nr_linear_layers}_aug={data_augmentation}"
                            f"_drop={dropout}_epochs:{args.nr_epochs}_lr:{lr}_gamma:{gamma}"
                        )
                        os.makedirs(model_folder, exist_ok=True)
                        save_path = os.path.join(model_folder, trained_network_file)
                        log(f"Model folder: {model_folder}")
                        log(f"Checkpoint path: {save_path}")

                        log(f"TRAIN CONFIG -> dropout={dropout}, lr={lr}, batch_size={batch_size}, "
                            f"dataset_size={dataset_size}, gamma={gamma}")
                        log(f"Num train samples: {len(train_sub)}  Num val samples: {len(val_ds)}")

                        best_val = float('inf')
                        patience = 20
                        epochs_no_improve = 0
                        best_epoch = 0
                        hist = []
                        t_global0 = time.perf_counter()

                        for epoch in range(args.nr_epochs):
                            if epochs_no_improve >= patience:
                                log(f"Early stopping: patience reached ({patience})")
                                break

                            log(f"===== EPOCH {epoch+1}/{args.nr_epochs} START (lr={current_lr(optimizer)}) =====")

                            # ------- Train -------
                            model.train()
                            train_loss_sum = 0.0
                            train_steps = 0
                            t_epoch0 = time.perf_counter()

                            with time_block(f"Train epoch {epoch+1}"):
                                for bidx, (xb, yb) in enumerate(train_loader, start=1):
                                    # Shapes and dtypes
                                    if bidx % PRINT_EVERY == 1:
                                        log(f"[Train] Batch {bidx}: xb.shape={tuple(xb.shape)}, xb.dtype={xb.dtype}, "
                                            f"yb.shape={tuple(yb.shape)}, yb.dtype={yb.dtype}")

                                    # Transfer
                                    t0 = time.perf_counter()
                                    xb = xb.to(device, non_blocking=True)   # (N,H,W,C) float32 0..255
                                    yb = yb.to(device, non_blocking=True)   # (N,) long
                                    t_transfer = time.perf_counter() - t0

                                    optimizer.zero_grad(set_to_none=True)

                                    # Forward + loss
                                    t0 = time.perf_counter()
                                    with autocast():
                                        logits = model(xb)  # model must accept HWC
                                        loss = loss_fn(logits, yb)
                                    t_forward = time.perf_counter() - t0

                                    # Backward + step
                                    t0 = time.perf_counter()
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                    t_backward = time.perf_counter() - t0

                                    train_loss_sum += float(loss.item())
                                    train_steps += 1

                                    if bidx % PRINT_EVERY == 0:
                                        log(f"[Train] Batch {bidx}: "
                                            f"transfer={t_transfer*1e3:.1f}ms "
                                            f"forward={t_forward*1e3:.1f}ms "
                                            f"backward+step={t_backward*1e3:.1f}ms "
                                            f"loss={float(loss.item()):.6f}")

                            scheduler.step()
                            train_loss = train_loss_sum / max(train_steps, 1)
                            t_epoch = time.perf_counter() - t_epoch0
                            log(f"[Train] epoch_loss={train_loss:.6f}  steps={train_steps}  epoch_time={t_epoch:.2f}s  lr->{current_lr(optimizer)}")

                            # ------- Validate -------
                            model.eval()
                            val_loss_sum = 0.0
                            val_steps = 0
                            with time_block(f"Validate epoch {epoch+1}"):
                                with torch.no_grad():
                                    for bidx, (xb, yb) in enumerate(val_loader, start=1):
                                        if bidx % VAL_PRINT_EVERY == 1:
                                            log(f"[Val] Batch {bidx}: xb.shape={tuple(xb.shape)}, yb.shape={tuple(yb.shape)}")
                                        t0 = time.perf_counter()
                                        xb = xb.to(device, non_blocking=True)
                                        yb = yb.to(device, non_blocking=True)
                                        with autocast():
                                            logits = model(xb)
                                            loss = loss_fn(logits, yb)
                                        val_loss_sum += float(loss.item())
                                        val_steps += 1
                                        if bidx % VAL_PRINT_EVERY == 0:
                                            log(f"[Val] Batch {bidx}: loss={float(loss.item()):.6f}, "
                                                f"batch_time={(time.perf_counter()-t0)*1e3:.1f}ms")

                            val_loss = val_loss_sum / max(val_steps, 1)
                            hist.append([train_loss, val_loss])

                            elapsed = time.perf_counter() - t_global0
                            eta = (elapsed / (epoch + 1)) * (args.nr_epochs - 1 - epoch)
                            log(f"Epoch {epoch+1:03d} SUMMARY  train={train_loss:.6f}  val={val_loss:.6f}  "
                                f"elapsed={elapsed:.1f}s  ETA=+{eta:.1f}s")

                            # Early stopping + save
                            if val_loss < best_val - 1e-8:
                                best_val = val_loss
                                best_epoch = epoch
                                epochs_no_improve = 0
                                torch.save(model.state_dict(), save_path)
                                log(f"[Checkpoint] Saved best at epoch {epoch+1}  val={best_val:.6f}")
                            else:
                                epochs_no_improve += 1
                                log(f"No improvement ({epochs_no_improve}/{patience})")

                            log(f"===== EPOCH {epoch+1}/{args.nr_epochs} END =====")

                        log(f"Best epoch: {best_epoch + 1} with validation loss: {best_val:.6f}")

                        # ---- Plot losses ----
                        train_losses = [x[0] for x in hist]
                        val_losses = [x[1] for x in hist]
                        with time_block("Plot & save curves"):
                            plt.figure()
                            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
                            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
                            plt.xlabel('Epochs')
                            plt.ylabel('Loss')
                            plt.title('Training and Validation Loss over Epochs')
                            plt.legend()
                            fig_path = os.path.join(model_folder, 'loss_plot.png')
                            plt.savefig(fig_path)
                            plt.close()
                            log(f"Saved plot to {fig_path}")

                        npy_path = os.path.join(model_folder, 'train_val_loss.npy')
                        np.save(npy_path, np.array(hist, dtype=np.float32))
                        log(f"Saved losses to {npy_path}")

    log("train() finished")
