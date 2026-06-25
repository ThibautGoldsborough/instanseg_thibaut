import os
# The loss allocates wildly varying tensor sizes per iteration (instance/seed/crop
# counts vary ~1000x across images), which fragments the CUDA caching allocator and
# makes RESERVED memory creep up across an epoch -> mid-epoch OOM even when peak
# *allocated* fits. expandable_segments lets the allocator grow/shrink segments
# instead, largely eliminating that fragmentation. setdefault so a user value wins.
# Must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# BEFORE any matplotlib import — silences Tk teardown errors under DDP.
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
from torch import nn
import torch
import torch.optim as optim
import argparse
from pathlib import Path
import pandas as pd

#torch.autograd.set_detect_anomaly(True) #For debugging cuda errors
parser = argparse.ArgumentParser()

#basic usage
parser.add_argument("-d_p", "--data_path", type=str, default=r"../datasets", help="Path to the .pth file")
parser.add_argument("-data", "--dataset", type=str, default="segmentation", help="Name of the dataset to load")
parser.add_argument("-zarr", "--zarr_root", type=str, default=None, help="Root of the zarr dataset (manifest.parquet lives here). Defaults to <data_path>/zarr. Built once from the .pth sources if missing.")
parser.add_argument('-source', '--source_dataset', default="all", type=str, help = "Which datasets to use for training. Input is 'all' or a list of datasets (e.g. [TNBC_2018,LyNSeC,IHC_TMA,CoNSeP]). Append a per-source relative sampling weight as name:w or name(w) (default 1, need not sum to 1), e.g. [dsb_2018:0.1,open-ai] — mainly affects sampling with -w True.")
parser.add_argument("-m_f", "--model_folder", type=str, default=None, help = "Name of the model to resume training. This must be a folder inside model_path")
parser.add_argument("-m_p", "--model_path", type=str, default=r"../models", help = "Path to the folder containing the models")
parser.add_argument("-o_p", "--output_path", type=str, default=r"../models", help = "Path to the folder where the results will be saved")
parser.add_argument("-e_s", "--experiment_str", type=str, default="my_first_instanseg", help = "String to identify the experiment")
parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('-num_workers', '--num_workers', default=3, type=int, help = "Number of CPU cores to use for data loading")
parser.add_argument('-prefetch_factor', '--prefetch_factor', default=2, type=int, help = "Number of batches each DataLoader worker prefetches ahead. Higher hides per-item augmentation/IO variance (the per-iter speed sawtooth) at the cost of RAM. Ignored when num_workers=0.")
parser.add_argument('-persistent_workers', '--persistent_workers', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Keep DataLoader workers alive across epochs (avoids the slow per-epoch worker respawn). WARNING: with the in-memory-array dataset this leaks RSS via copy-on-write and can OOM after ~17 epochs — see data_loader.py.")
parser.add_argument('-ci', '--channel_invariant', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to add a channel invariant model to the pipeline")
parser.add_argument('-target', '--target_segmentation', default="N",type=str, help = " Cells or nuclei or both? Accepts: C,N, NC")  
parser.add_argument('-pixel_size', '--requested_pixel_size', default=None, type=float, help = "Requested pixel size to rescale the input images")
parser.add_argument('--force_pseudo_pixel_size', default=False, type=lambda x: (str(x).lower() == 'true'), help = "If True, ignore pixel_size from image metadata and always estimate it from average instance area.")

#advanced usage
parser.add_argument("-bs", "--batch_size", type=int, default=3)
parser.add_argument('-find_bs', '--find_batch_size', default=False, type=lambda x: (str(x).lower() == 'true'), help="Before training, probe the largest batch size that fits in GPU memory (fwd+bwd, weights untouched) and use it. Overrides --batch_size. DDP-safe (min across ranks).")
parser.add_argument('-max_bs', '--max_batch_size', default=512, type=int, help="Upper cap for --find_batch_size probing.")
parser.add_argument('-bs_mem_frac', '--bs_mem_fraction', default=0.8, type=float, help="Fraction of GPU memory --find_batch_size targets. Lower (e.g. 0.8) for more headroom against DDP overhead / worst-case-density batches.")
parser.add_argument("-e", "--num_epochs", type=int, default=500)
parser.add_argument('-len_epoch', '--length_of_epoch', default=1000, type=int, help = "Number of training samples per epoch")
parser.add_argument('-len_eval', '--length_of_eval', default=200, type=int, help = "Number of validation samples per epoch")
parser.add_argument("-lr", "--lr", type=float, default=0.001, help = "Learning rate")
parser.add_argument('-lr_scaling', '--lr_scaling', default="sqrt", type=str, choices=["none", "sqrt", "linear"], help="Scale --lr by the effective (global) batch size relative to --base_batch_size: lr *= (eff_batch/base)^p, p=0.5 (sqrt, recommended for AdamW) or 1.0 (linear), 'none' disables. Applied after --find_batch_size resolves the batch. Existing warmup is kept.")
parser.add_argument('-base_bs', '--base_batch_size', default=10, type=int, help="Reference (global) batch size at which --lr is the tuned value, for --lr_scaling. Default 10 (i.e. lr=1e-3 @ batch 10).")
parser.add_argument("-optim", "--optimizer", type=str, default="adamw", help = "Optimizer to use, adam, sgd or adamw")
parser.add_argument("-m", "--model_str", type=str, default="maxvit_base", help = "Model backbone to use")
parser.add_argument("-s", "--save", type=bool, default=True, help = "Whether to save model outputs every time a new best F1 score is achieved")
parser.add_argument("-l_fn", "--loss_function", type=str, default='instanseg_loss', help = "Method to use for segmentation, only instanseg_loss is supported")
parser.add_argument("-n_sigma", "--n_sigma", type=int, default=2, help = "Number of sigma channels, must be at least 1")
parser.add_argument("-cluster", "--on_cluster", type=bool, default=False, help ="Flag to disable tqdm progress bars and other non-essential outputs, useful for running on a cluster")
parser.add_argument("-w", "--weight", default=False, type = lambda x: (str(x).lower() == 'true'), help = "Weight the random sampler in the training set to oversample images with more instances")
parser.add_argument("-layers", "--layers", type=str, default="[32, 64, 128, 256]", help = "UNet layers")
parser.add_argument("-slice", "--data_slice", type=int, default=None, help = "Slice of the dataset to use, useful for debugging (e.g. only train on 1 image)")
parser.add_argument("-clip", "--clip", type=float, default=20, help ="Gradient clipping value")
parser.add_argument("-decay", "--weight_decay", type=float, default=0.000, help = "Weight decay")
parser.add_argument("-drop", "--dropprob", type=float, default=0., help = "Dropout probability applied to encoder/decoder blocks (Dropout2d)")
parser.add_argument("-drop_path", "--drop_path_rate", type=float, default=0., help = "Stochastic depth rate. Applied uniformly to every MaxViT block (paper-style fixed rate).")
parser.add_argument("-tf", "--transform_intensity", type=float, default=0.5, help = "Intensity transformation factor")
parser.add_argument("-dim_in", "--dim_in", type=int, default=3,help="Number of channels that the (backbone) model expects. This is also the number of channels a channel invariant model would output.")
parser.add_argument("-dummy", "--dummy", default=False, type=lambda x: (str(x).lower() == 'true'),help="Use the training set as a validation set, this will trigger a warning message. use only for debugging")
parser.add_argument('-bg_weight', '--bg_weight', default=None, type= float, help = "Weight to assign to the background class in the loss function")
parser.add_argument('-instance_loss_fn', '--instance_loss_fn', default="lovasz_hinge", type=str, help = "Loss function to use for instance segmentation: lovasz_hinge or dice_loss are supported. lovasz_hinge is a lot slower to start converging")
parser.add_argument('-seed_loss_fn', '--seed_loss_fn', default="l1_distance", type=str, help = "Loss function to use for seed selection: ce, l1_distance, l2_distance, or l1_poisson. ce is much faster, but l1_distance is usually more accurate. l1_poisson uses Poisson field instead of EDT.")
parser.add_argument('-mask_loss_fn', '--mask_loss_fn', default=None, type=str, help = "Loss function to use for the mask channel when dim_seeds=2: ce or dice are supported. If set, dim_seeds is forced to 2.")
parser.add_argument('-anneal', '--cosineannealing', default=True, type=lambda x: (str(x).lower() == 'true'), help = "Cosine-anneal the LR over --num_epochs down to lr*1e-2 (ON by default; pairs with --warmup_epochs as warmup->cosine). Decay is paced to --num_epochs, so set -e to your real training length. Set False for a constant post-warmup LR. No effect on SAM/DINO (they use their own multi-stage schedule).")
parser.add_argument('-warmup', '--warmup_epochs', default=10, type=int, help = "Linear LR warmup epochs, prepended to the schedule for ANY model (0 = off; default 10, on). Recommended when --lr_scaling raises the LR for large batches. Composes with --cosineannealing; for SAM/DINO models it sets the warmup length of their multi-stage schedule.")
parser.add_argument('-warmup_phase', '--warmup_phase', default="hotstart", type=str, choices=["main", "hotstart"], help = "Where the --warmup_epochs ramp lives: 'hotstart' (default) ramps DURING the hotstart phase (then main starts at full LR); 'main' ramps the first epochs of MAIN training. 'hotstart' falls back to 'main' if --hotstart_training is 0.")
parser.add_argument('-o_h', '--optimize_hyperparameters', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to optimize hyperparameters every 10 epochs")
parser.add_argument('-hotstart', '--hotstart_training', default=10, type=int, help = "Number of epochs to train the model with ce before starting the main training loop (default=10)")
parser.add_argument('-window', '--window_size', default=128, type=int, help = "Size of the window containing each instance")
parser.add_argument('-multihead', '--multihead', default= False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to branch the decoder into multiple heads.")
parser.add_argument('-adaln', '--adaln', default=False, type=lambda x: (str(x).lower() == 'true'), help = "MaxViT only: condition a single-head model on cell-vs-nucleus via AdaLN. Each training sample picks one available label map (C or N) at random and passes it to the model as a condition. Requires a maxvit backbone.")
parser.add_argument('-mae', '--mae', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Masked-image-modeling (MAE-style) self-supervised pretraining: mask input patches and reconstruct them. Reuses the full data/aug/schedule pipeline; replaces the segmentation loss. Skips hotstart and F1 eval.")
parser.add_argument('-mae_mask_ratio', '--mae_mask_ratio', default=0.6, type=float, help = "MAE: fraction of input patches to mask (default 0.6).")
parser.add_argument('-mae_patch_size', '--mae_patch_size', default=16, type=int, help = "MAE: patch size for masking and the reconstruction loss (must divide the crop size).")
parser.add_argument('-mae_norm_target', '--mae_norm_target', default=True, type=lambda x: (str(x).lower() == 'true'), help = "MAE: normalize each target patch to zero-mean/unit-var before the loss (MAE-paper default).")
parser.add_argument('-mae_init', '--mae_init', default="", type=str, help = "Fine-tune from an MAE checkpoint: path to its folder or model_weights.pth. Transfers backbone (encoder+decoder) weights into a fresh segmentation model (strict=False, shape-filtered: reconstruction heads + pixel_classifier stay random). Mutually exclusive with --model_folder and --mae.")
parser.add_argument('-backbone_lr_scale', '--backbone_lr_scale', default=1.0, type=float, help = "Multiply the LR of the pretrained backbone (encoder/decoder) params by this factor; e.g. 0.1 for discriminative fine-tuning from --mae_init. 1.0 = uniform LR (no param groups).")
parser.add_argument('-dim_coords', '--dim_coords', default=2, type=int, help = "Dimensionality of the coordinate system. Little support for anything but 2")
parser.add_argument('-dim_seeds', '--dim_seeds', default=1, type=int, help = "Number of seed maps to produce. Little support for anything but 1")
parser.add_argument('-norm', '--norm', default="BATCH", type=str, help = "Norm layer to use: None, INSTANCE, INSTANCE_INVARIANT, BATCH")
parser.add_argument('-mlp_w', '--mlp_width', default=5, type=int, help = "Width of the MLP hidden dim")
parser.add_argument('-augmentation_type', '--augmentation_type', default="minimal", type=str, help = "'minimal' or 'heavy' or 'brightfield_only'")
parser.add_argument('-use_instance_channels', '--use_instance_channels', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Enable the add_instance_channels augmentation (appends binary masks of random GT instances as extra channels). Only active for channel-invariant models.")
parser.add_argument('-show_augmentations', '--show_augmentations', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Save 10 example augmented image+label pairs to the output folder")
parser.add_argument('-adaptor_net', '--adaptor_net_str', default="1", type=str, help = "Adaptor net to use")
parser.add_argument('-freeze', '--freeze_main_model', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to freeze the main model")
parser.add_argument('-f_e', '--feature_engineering', default="0", type=str, help = "Feature engineering function to use")
parser.add_argument("-f","--f", default = None, type = str, help = "ignore, this is for jypyter notebook compatibility")
parser.add_argument('-rng_seed', '--rng_seed', default=None, type=int, help = "Optional seed for the random number generator")
parser.add_argument('-use_deterministic', '--use_deterministic', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to use deterministic algorithms (default=False)")
parser.add_argument('-tile', '--tile_size', default=256, type=int, help = "Tile sizes for the input images")
parser.add_argument('-modality', '--modality_filter', default=None, type=str, help = "Filter datasets by image modality (e.g. 'Brightfield', 'Fluorescence'). Default None uses all modalities.")
parser.add_argument('-cluster_sample', '--cluster_sample', default=0, type=int, help="Cluster train images with the model's own embeddings after hotstart and weight the sampler by cluster frequency. 0 = off; N > 0 = refresh clusters every N main-training epochs.")
parser.add_argument('-lora_rank', '--lora_rank', default=0, type=int, help="LoRA rank for SAM/DINO backbone. 0 = disabled, 16 is a good default.")
parser.add_argument('-fp16', '--fp16', default=False, type=lambda x: (str(x).lower() == 'true'), help="Enable mixed precision (float16) training")
parser.add_argument('-seed_merging', '--seed_merging', default=False, type=lambda x: (str(x).lower() == 'true'), help="Enable seed-seed attention merging")
parser.add_argument('-uncertainty_weighting', '--uncertainty_weighting', default=False, type=lambda x: (str(x).lower() == 'true'), help="Learn task weights via uncertainty (Kendall et al. 2018)")
parser.add_argument('-batched_instance_loss', '--batched_instance_loss', default=True, type=lambda x: (str(x).lower() == 'true'), help="Compute instance loss in one batched pass (True) or per-image (False)")
parser.add_argument('-preemptable', '--preemptable', default=False, type=lambda x: (str(x).lower() == 'true'), help="Enable preemption-safe training: saves full training state and auto-resumes from checkpoint if preempted on SLURM")
parser.add_argument('-preempt_interval', '--preempt_save_interval', default=10, type=int, help="Save preemptable checkpoint every N epochs (default=1). Higher values reduce I/O for large models at the cost of losing more progress on preemption.")
parser.add_argument('-skip_bad_batches', '--skip_bad_batches', default=True, type=lambda x: (str(x).lower() == 'true'), help="Skip training batches whose loss or gradient norm is non-finite (NaN/Inf) and warn. Default=True.")
parser.add_argument('-reset_optimizer', '--reset_optimizer', default=False, type=lambda x: (str(x).lower() == 'true'), help="When hotstarting from --model_folder, start with a fresh optimizer (discard the saved optimizer state/moments) instead of restoring it. Use when finetuning onto a new task/dataset where the old Adam moments describe a different objective. Default=False (restore optimizer state). No effect on LoRA / warm-started NC heads / channel-invariant runs, which always reset.")

_bool = lambda x: (str(x).lower() == 'true')
parser.add_argument('-compile', '--compile', default=False, type=_bool, help="Whether to torch.compile the model")
parser.add_argument('-compile_mode', '--compile_mode', default="default", type=str, help="torch.compile mode: default, reduce-overhead, max-autotune")
parser.add_argument('-time_dataloading', '--time_dataloading', default=False, type=_bool, help="Print a per-epoch timing breakdown (dataload wait vs GPU compute) to diagnose dataloader bottlenecks. Adds per-iter cuda syncs (slightly slows the run).")
parser.add_argument('-shard_dataset_per_rank', '--shard_dataset_per_rank', default=True, type=_bool, help="Per-rank dataset sharding under multi-GPU DDP. Each rank sees 1/N of the data; total samples per epoch is preserved by scaling length_of_epoch.")


def _launcher_cmdline() -> str | None:
    """Best-effort: the launcher invocation (e.g. ``accelerate launch ...``,
    ``torchrun``) by walking up the parent-process chain via /proc (Linux only).
    accelerate/torchrun strip their own flags from sys.argv, so this is the only
    way to record the full command. Returns None if not found / not on Linux."""
    import shlex
    try:
        pid = os.getppid()
        for _ in range(8):
            if pid <= 1:
                return None
            parts = [p.decode() for p in
                     Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00") if p]
            if any(k in " ".join(parts) for k in
                   ("accelerate", "torchrun", "torch.distributed", "deepspeed")):
                return " ".join(shlex.quote(p) for p in parts)
            status = Path(f"/proc/{pid}/status").read_text()
            pid = int(next(l for l in status.splitlines()
                           if l.startswith("PPid:")).split()[1])
    except Exception:
        return None
    return None


def save_run_command(output_path) -> None:
    """Append the command used to launch this run to ``output_path/command.txt``
    (the python invocation from sys.argv + a best-effort launcher line). Appends
    so re-launches / preemptable resumes accumulate a history."""
    import shlex, socket
    from datetime import datetime
    py_cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    lines = [
        f"# {datetime.now().isoformat(timespec='seconds')}  host={socket.gethostname()}",
        f"# cwd: {os.getcwd()}",
    ]
    launcher = _launcher_cmdline()
    if launcher:
        lines.append(f"# launcher: {launcher}")
    lines.append(py_cmd)
    try:
        with open(Path(output_path) / "command.txt", "a") as f:
            f.write("\n".join(lines) + "\n\n")
    except Exception as e:
        print(f"[command.txt] could not save run command: {e}")


def save_training_plot(train_losses, test_losses, f1_list, f1_list_cells, output_path, cells_and_nuclei=False, hotstart_epoch=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    window_size = max(len(train_losses) // 10 + 1, 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    clip_val = np.percentile(test_losses, [99])
    test_clipped = np.clip(test_losses, 0, clip_val[0])
    clip_val = np.percentile(train_losses, [99])
    train_clipped = np.clip(train_losses, 0, clip_val[0])

    ax1.plot(np.convolve(train_clipped, np.ones(window_size), 'valid') / window_size, label="train loss", color="tab:blue")
    ax1.plot(np.convolve(test_clipped, np.ones(window_size), 'valid') / window_size, label="test loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    if hotstart_epoch is not None and hotstart_epoch > 0:
        ax1.axvline(x=hotstart_epoch, color='red', linestyle=':', linewidth=1, label="hotstart end")

    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1)

    if cells_and_nuclei:
        ax2.plot(f1_list, label="f1 nuclei", color="tab:green", linestyle="--")
        ax2.plot(f1_list_cells, label="f1 cells", color="tab:red", linestyle="--")
    else:
        ax2.plot(f1_list, label="f1 score", color="tab:green", linestyle="--")
    ax2.legend(loc="upper right")

    fig.savefig(output_path / "training_plot.png")
    plt.close(fig)


def _build_scheduler(optimizer, args, warmup=None):
    """Per-epoch LR scheduler with optional linear warmup (--warmup_epochs) for ANY
    backbone, composed in front of the base schedule. Returns None for the plain
    constant-LR + no-warmup case (preserves the legacy default).

    ``warmup`` overrides the warmup length (else uses args.warmup_epochs); pass 0 to
    suppress warmup in the main phase when it's been placed in hotstart instead.

    - SAM/DINO models keep their multi-stage schedule (warmup + plateau + 2 decays);
      warmup length = warmup or 10 (back-compat default there).
    - else --cosineannealing -> cosine, optionally prefixed by a linear warmup.
    - else -> linear warmup then constant, or None when warmup is off.
    """
    from torch.optim.lr_scheduler import (LambdaLR, CosineAnnealingLR, LinearLR, SequentialLR)
    warmup = max(0, int(args.warmup_epochs if warmup is None else warmup))
    max_epochs = max(1, args.num_epochs)
    sam_like = args.model_str.lower() in ("cellposesam", "instanseg_sam", "instanseg_dino")

    if sam_like:
        we = warmup or 10
        def lr_schedule(epoch):
            if epoch < we:
                return (epoch + 1) / we
            elif epoch < max_epochs - 150:
                return 1.0
            elif epoch < max_epochs - 50:
                return 0.1
            return 0.01
        return LambdaLR(optimizer, lr_lambda=lr_schedule)

    if args.cosineannealing:
        cos = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup), eta_min=args.lr * 1e-2)
        if warmup > 0:
            warm = LinearLR(optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup)
            return SequentialLR(optimizer, [warm, cos], milestones=[warmup])
        return cos

    if warmup > 0:  # linear warmup then constant
        return LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / warmup))
    return None


def main(model, loss_fn, train_loader, test_loader, num_epochs=1000, epoch_name='output_epoch',
         prior_train_losses=None, prior_test_losses=None, prior_f1_list=None, prior_f1_list_cells=None, hotstart_epoch=None,
         scaler=None, start_epoch=0, prior_best_f1=-1,
         resample_fn=None, resample_interval=0):
    from instanseg.utils.AI_utils import optimize_hyperparameters, train_epoch, test_epoch
    global best_f1_score, device, method, iou_threshold, args, optimizer, scheduler

    accelerator = getattr(args, 'accelerator', None)
    is_main = accelerator.is_main_process if accelerator is not None else True

    train_losses = list(prior_train_losses) if prior_train_losses else []
    test_losses = list(prior_test_losses) if prior_test_losses else []

    best_f1_score = prior_best_f1
    f1_list = list(prior_f1_list) if prior_f1_list else []
    f1_list_cells = list(prior_f1_list_cells) if prior_f1_list_cells else []

    for epoch in range(start_epoch, num_epochs):

        if resample_fn is not None and resample_interval > 0 and epoch > 0 and epoch % resample_interval == 0:
            if is_main:
                print(f"[resample] Refreshing Leiden clusters at epoch {epoch}")
            train_loader = resample_fn()

        if is_main:
            print("Epoch:", epoch)

        train_loss, train_time = train_epoch(model, device, train_loader, loss_fn, optimizer, args=args, scaler=scaler)

        if epoch <= 5 and not args.model_folder and start_epoch == 0:  # Training is just starting AND we are not resuming
            save_epoch_outputs = True
        else:
            save_epoch_outputs = False

        test_loss, f1_score, test_time = test_epoch(model, device, test_loader, loss_fn, debug=False,
                                                    best_f1=best_f1_score,
                                                    save_bool=save_epoch_outputs,
                                                    args=args,
                                                    postprocessing_fn=(method.postprocessing if method is not None else None),
                                                    method=method,
                                                    iou_threshold=iou_threshold,
                                                    use_amp=scaler is not None,
                                                    save_str=str(
                                                        args.output_path / str(
                                                            f"epoch_outputs/{epoch_name}_" + str(epoch))))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 ==0 and args.optimize_hyperparameters and method is not None:
            best_params = optimize_hyperparameters(model,postprocessing_fn = method.postprocessing,data_loader= test_loader,verbose = not args.on_cluster, show_progressbar = not args.on_cluster)
            method.update_hyperparameters(best_params)


        dict_to_print = {"train_loss": train_loss, "test_loss": test_loss, "training_time": int(train_time),
                         "testing_time": int(test_time)}

        if hasattr(method, 'last_seed_loss'):
            # last_*_loss are detached tensors (de-synced in the loss); coerce
            # to float here — this once-per-epoch read is the only sync needed.
            dict_to_print["seed_loss"] = float(method.last_seed_loss)
            dict_to_print["instance_loss"] = float(method.last_instance_loss)
        if method is not None and method.uncertainty_weighting:
            import math
            dict_to_print["w_seed"] = math.exp(-method.log_var_seed.item())
            dict_to_print["w_inst"] = math.exp(-method.log_var_inst.item())

        if args.mae:
            # No F1 for reconstruction; monitor the (negative) val loss so the
            # best-checkpoint logic below saves the lowest-loss model.
            f1_list.append(float('nan'))
            dict_to_print["recon_loss"] = test_loss
            f1_score = -test_loss
        elif args.dual_head_output:
            f1_list.append(f1_score[0])
            f1_list_cells.append(f1_score[1])
            dict_to_print["f1_score_nuclei"] = f1_score[0]
            dict_to_print["f1_score_cells"] = f1_score[1]
            f1_score = np.nanmean(f1_score)
            dict_to_print["f1_score_joint"] = f1_score

        else:
            f1_score = f1_score[0]
            f1_list.append(f1_score)
            dict_to_print["f1_score"] = f1_score

        if scheduler is not None:
            dict_to_print["lr:"] = optimizer.param_groups[0]["lr"]
            scheduler.step()

                
        _is_new_best = f1_score > best_f1_score
        if _is_new_best or save_epoch_outputs:
            best_f1_score = np.maximum(f1_score, best_f1_score)
            if is_main:
                print("Saving model, best f1_score:", best_f1_score)

        _should_save_best = _is_new_best or save_epoch_outputs
        _should_save_preempt = args.preemptable and (epoch % args.preempt_save_interval == 0 or epoch == num_epochs - 1)
        if (_should_save_best or _should_save_preempt) and is_main:
            _model_state = (accelerator.unwrap_model(model).state_dict()
                            if accelerator is not None else model.state_dict())
            _optim_state = optimizer.state_dict()
            _sched_state = scheduler.state_dict() if scheduler is not None else None

            if _should_save_best:
                best_checkpoint = {
                    'f1_score': float(best_f1_score),
                    'epoch': int(epoch),
                    'model_state_dict': _model_state,
                    'optimizer_state_dict': _optim_state,
                }
                if _sched_state is not None:
                    best_checkpoint['scheduler_state_dict'] = _sched_state
                torch.save(best_checkpoint, args.output_path / "model_weights.pth")

            if _should_save_preempt:
                resume_state = {
                    'f1_score': float(best_f1_score),
                    'epoch': int(epoch),
                    'model_state_dict': _model_state,
                    'optimizer_state_dict': _optim_state,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'f1_list': f1_list,
                    'f1_list_cells': f1_list_cells,
                    'epoch_name': epoch_name,
                }
                if _sched_state is not None:
                    resume_state['scheduler_state_dict'] = _sched_state
                # Atomic save: write to temp file then rename, so SIGTERM during
                # write can't corrupt the checkpoint and break resume
                _tmp_path = args.output_path / "preemptable_state.pth.tmp"
                torch.save(resume_state, _tmp_path)
                _tmp_path.rename(args.output_path / "preemptable_state.pth")


        if is_main:
            print(", ".join(f"{k}: {v:.5g}" for k, v in dict_to_print.items()))
            save_training_plot(train_losses, test_losses, f1_list, f1_list_cells,
                               args.output_path, args.dual_head_output, hotstart_epoch=hotstart_epoch)

    return model, train_losses, test_losses, f1_list, f1_list_cells

from typing import Dict
def instanseg_training(segmentation_dataset: Dict = None, **kwargs):

    global device, method, iou_threshold, args, optimizer, scheduler
    args = parser.parse_args()

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Argument {key} not recognized")


    from instanseg.utils.utils import plot_average, _choose_device
    from instanseg.utils.model_loader import build_model_from_dict, load_model_weights
    from instanseg.utils.zarr_loader import get_zarr_loaders, ensure_zarr_dataset

    # Accelerator setup (replaces nn.DataParallel). Initialize first so is_main_process
    # is available for guarding mkdir / prints / saves throughout.
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from datetime import timedelta

    # static_graph=True is REQUIRED: the loss reuses pixel_classifier once per
    # instance per image in a single backward (instanseg_loss.py:1453/1483) —
    # "reused parameters", which only static_graph supports (find_unused_parameters
    # raises "marked ready twice"). It is also valid under --adaln: AdaLN's
    # grad-participation set is static, NOT dynamic — the shared nn.Embedding is
    # dense so every row gets a (possibly zero) gradient whatever class the batch
    # picks, and the per-site proj Linears apply FiLM on every forward. So no
    # find_unused_parameters branch (it was unnecessary and broke the reuse).
    ddp_kwargs = DistributedDataParallelKwargs(
        static_graph=True,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    # Raise the NCCL collective timeout well above the 600s default: with --compile
    # (per-rank compile of the probe + first graph can take many minutes and varies
    # across ranks) and the one-off rank-0 zarr dataset build, a straggler can lag
    # past 10min and trip the watchdog on the prepare() param broadcast.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=120))
    accelerator = Accelerator(
        mixed_precision="bf16" if args.fp16 else "no",
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )
    is_main = accelerator.is_main_process
    args.accelerator = accelerator   # consumed by train_epoch / test_epoch

    args.data_path = Path(args.data_path)

    if is_main and not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    accelerator.wait_for_everyone()

    args.output_path = Path(args.output_path) / args.experiment_str
    if is_main:
        print("Saving results to {}".format(os.path.abspath(args.output_path)))
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        save_run_command(args.output_path)
    accelerator.wait_for_everyone()

    # Preemptable: detect existing checkpoint for auto-resume
    _preemptable_resuming = False
    _preemptable_checkpoint = None
    if args.preemptable and not args.model_folder:
        _ckpt_path = args.output_path / "preemptable_state.pth"
        if _ckpt_path.exists():
            if is_main:
                print(f"[preemptable] Found existing checkpoint at {_ckpt_path}, will auto-resume")
            _preemptable_resuming = True
            _preemptable_checkpoint = torch.load(_ckpt_path, weights_only=False, map_location=accelerator.device)
        elif is_main:
            print("[preemptable] No existing checkpoint found, starting fresh")

    os.environ["INSTANSEG_DATASET_PATH"] = os.environ.get("INSTANSEG_DATASET_PATH", str(args.data_path))
    os.environ["INSTANSEG_OUTPUT_PATH"] = os.environ.get("INSTANSEG_OUTPUT_PATH", str(args.output_path))

    # Seed as many rngs as we can
    if args.rng_seed:
        if is_main:
            print(f'Setting RNG seed to {args.rng_seed}')
        torch.manual_seed(args.rng_seed)
        np.random.seed(args.rng_seed)
        import random
        random.seed(args.rng_seed)
    elif is_main:
        print('RNG seed not set')

    if args.use_deterministic:
        if is_main:
            print('Setting use_deterministic_algorithms=True')
        torch.use_deterministic_algorithms(True,warn_only = True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


    args.layers = eval(args.layers)
    args_dict = vars(args)
    num_epochs = args.num_epochs
    n_sigma = args.n_sigma
    dim_in = args.dim_in

    if args.norm == "None":
        args.norm = None

    if len(args.target_segmentation) == 2:
        args.cells_and_nuclei = True
    else:
        args.cells_and_nuclei = False

    # AdaLN conditioning makes the model single-head (it predicts whichever of
    # C/N the per-sample condition selects), so model output, loss, and F1
    # metrics are single-channel even though the data still carries both label
    # maps. `dual_head_output` is the effective "two channels out" flag.
    if args.adaln:
        if not args.model_str.lower().startswith(("maxvit", "maxxvit")):
            raise ValueError(f"--adaln requires a maxvit backbone, got model_str={args.model_str!r}")
        # Conditioning is cell-vs-nucleus, so both label maps must be loaded.
        # With a single target only one class is ever sampled and the other
        # embedding never trains, silently making --adaln a no-op. Require NC.
        if not ("N" in args.target_segmentation.upper() and "C" in args.target_segmentation.upper()):
            raise ValueError(
                f"--adaln conditions on cell-vs-nucleus and needs both label maps; "
                f"pass `-target NC` (got -target {args.target_segmentation!r}). "
                f"With a single target only one condition is ever sampled and the "
                f"other class embedding never trains."
            )
    args.dual_head_output = args.cells_and_nuclei and not args.adaln

    if args.mae:
        # MAE reconstructs a fixed-channel image: no conditioning / channel-invariance / dual head.
        if args.adaln:
            raise ValueError("--mae is incompatible with --adaln.")
        if args.channel_invariant:
            raise ValueError("--mae requires a fixed input-channel backbone; --channel_invariant is not supported.")
        if not args.model_str.lower().startswith(("maxvit", "maxxvit")):
            raise ValueError(f"--mae currently supports maxvit backbones only, got model_str={args.model_str!r}")
        if args.hotstart_training > 0:
            # Hotstart swaps in seg losses MAE doesn't use; warn+zero so a plain --mae run works.
            import warnings
            warnings.warn(f"--mae does not use hotstart; forcing --hotstart_training 0 (was {args.hotstart_training}).")
            args.hotstart_training = 0
        args.cells_and_nuclei = False
        args.dual_head_output = False

    if args.mae_init:
        if args.mae:
            raise ValueError("--mae_init fine-tunes a pretrained MAE model for segmentation; don't combine with --mae.")
        if args.model_folder:
            raise ValueError("--mae_init and --model_folder are mutually exclusive (warm-start from MAE vs resume a seg checkpoint).")

    device = accelerator.device
    args.device = device

    if args.mae:
        # No InstanSeg loss module: backbone reconstructs the input (dim_out=dim_in).
        from functools import partial
        from instanseg.utils.loss.mae import mae_loss_fn
        method = None
        dim_out = int(dim_in)
        loss_fn = partial(mae_loss_fn, patch_size=args.mae_patch_size, norm_target=args.mae_norm_target)

    elif args.loss_function == "instanseg_loss":
        from instanseg.utils.loss.instanseg_loss import InstanSeg


        if args.mask_loss_fn is not None and args.dim_seeds != 2:
            import warnings
            warnings.warn(f"mask_loss_fn='{args.mask_loss_fn}' requires dim_seeds=2, setting dim_seeds=2")
            args.dim_seeds = 2
        if args.dim_seeds == 2 and args.mask_loss_fn is None:
            import warnings
            warnings.warn("dim_seeds=2 but mask_loss_fn not set, defaulting to mask_loss_fn='dice'")
            args.mask_loss_fn = "dice"

        method = InstanSeg(instance_loss_fn_str=args.instance_loss_fn,
                        seed_loss_fn = args.seed_loss_fn,
                        device = device,
                        n_sigma=n_sigma,
                        cells_and_nuclei=args.dual_head_output,
                        window_size = args.window_size,
                        dim_coords= args.dim_coords,
                        dim_seeds = args.dim_seeds,
                        feature_engineering_function=args.feature_engineering,
                        bg_weight = args.bg_weight,
                        mask_loss_fn = args.mask_loss_fn,
                        seed_merging = args.seed_merging,
                        uncertainty_weighting = args.uncertainty_weighting,
                        batched_instance_loss = args.batched_instance_loss)

        def loss_fn(*args, **kwargs):
            return method.forward(*args, **kwargs)

        dim_out = method.dim_out

    else:
        raise NotImplementedError("Loss function not recognized", args.loss_function)

    args.dim_out = dim_out
    args_dict = vars(args)

    if int(dim_in) == 0:
        args_dict["dim_in"] = None
    else:
        args_dict["dim_in"] = int(dim_in)
    args_dict["dropprob"] = float(args.dropprob)
    args_dict["drop_path_rate"] = float(args.drop_path_rate)

    model = build_model_from_dict(args_dict, random_seed=args.rng_seed)

    if args.mae:
        # Wrap backbone into a masked-reconstruction model (param-free masking).
        from instanseg.utils.loss.mae import MAEWrapper
        model = MAEWrapper(model, patch_size=args.mae_patch_size, mask_ratio=args.mae_mask_ratio)

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, torch.randn(1,3,256,256))
        if is_main:
            print("Number of flops:",flops.total()/1e9)
        #from fvcore.nn import flop_count_str
       # print(flop_count_str(flops))
    except:
        pass

    def get_optimizer(parameters, args, lr=None):
        lr = lr if lr is not None else args.lr
        if args.optimizer.lower() == "adam":
            return optim.Adam(parameters, lr=lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "sgd":
            return optim.SGD(parameters, lr=lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "adamw":
            return optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "adopt":
            from adopt import ADOPT
            return ADOPT(parameters, lr=lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError("Optimizer not recognized", args.optimizer)

    def trainable_params(model):
        # Discriminative LR: backbone (encoder/decoder) at lr*backbone_lr_scale, rest
        # (heads/pixel_classifier) at base lr. Scale 1.0 => plain iterator (no change).
        if args.backbone_lr_scale == 1.0:
            return filter(lambda p: p.requires_grad, model.parameters())
        bb, rest = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                (bb if (".encoder." in f".{n}" or ".decoders." in f".{n}") else rest).append(p)
        return [{"params": bb, "lr": args.lr * args.backbone_lr_scale}, {"params": rest}]

    if args.loss_function in ["instanseg_loss"] and not args.mae:
        from instanseg.utils.loss.instanseg_loss import has_pixel_classifier_model

        if not has_pixel_classifier_model(model):
            model = method.initialize_pixel_classifier(model, MLP_width = args.mlp_width)

    # Warm-start backbone from an MAE checkpoint (before channel-invariant wrapping).
    if args.mae_init:
        from instanseg.utils.loss.mae import load_mae_backbone
        _mae_path = Path(args.mae_init)
        if not _mae_path.exists():
            _mae_path = Path(args.model_path) / args.mae_init
        model = load_mae_backbone(model, _mae_path, verbose=is_main)

    if args.model_folder:
        if args.model_folder == "None":
            args.model_folder = ""

        model, model_dict = load_model_weights(model, path=args.model_path, folder=args.model_folder, device=device, dict = args_dict)

        # Enable LoRA before creating optimizer if needed
        if args.lora_rank > 0 and hasattr(model, 'enable_lora'):
            model.freeze_backbone()
            model.enable_lora(rank=args.lora_rank)
            model.unfreeze_backbone()

        if not args.channel_invariant:
            if args.lora_rank > 0 and hasattr(model, 'enable_lora'):
                optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
                # Skip loading optimizer state dict — LoRA changes parameter groups
                if is_main:
                    print("LoRA enabled: skipping optimizer state dict (parameter groups changed)")
            elif model_dict.get('duplicated_decoder_heads', False):
                optimizer = get_optimizer(model.parameters(), args)
                # Skip loading optimizer state dict — warm-starting an NC model
                # from a single-task checkpoint adds decoder heads, so the
                # parameter groups no longer match.
                if is_main:
                    print("Warm-started NC heads: skipping optimizer state dict "
                          "(parameter groups changed)")
            elif args.reset_optimizer:
                optimizer = get_optimizer(model.parameters(), args)
                # Explicitly requested fresh optimizer — discard saved moments.
                # Use when finetuning onto a new task/dataset where the old
                # Adam moments describe a different objective.
                if is_main:
                    print("--reset_optimizer: starting with a fresh optimizer "
                          "(discarded saved optimizer state)")
            else:
                optimizer = get_optimizer(model.parameters(),args)
                optimizer.load_state_dict(model_dict['optimizer_state_dict'])

        if is_main:
            print("Resuming training from epoch", model_dict['epoch'])

    else:
        # Enable LoRA before creating optimizer (when no hotstart to handle it)
        if args.lora_rank > 0 and args.hotstart_training == 0 and hasattr(model, 'enable_lora'):
            model.freeze_backbone()
            model.enable_lora(rank=args.lora_rank)
            model.unfreeze_backbone()
            optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
        else:
            optimizer = get_optimizer(trainable_params(model), args)

    if args.channel_invariant:
        from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
        if not has_AdaptorNet(model):
            model = AdaptorNetWrapper(model,adaptor_net_str=args.adaptor_net_str, norm = args.norm)

        if args.freeze_main_model == True:
            params = model.model.AdaptorNet.parameters()
        else:
            params = model.parameters()

        optimizer = get_optimizer(params, args)

    scheduler = _build_scheduler(optimizer, args)

    # Preemptable: restore model weights from checkpoint (optimizer/scheduler restored later, phase-specifically)
    if _preemptable_resuming and _preemptable_checkpoint is not None:
        _resume_epoch = _preemptable_checkpoint['epoch'] + 1
        _ckpt_epoch_name = _preemptable_checkpoint.get('epoch_name', 'output_epoch')
        _was_in_hotstart = (_ckpt_epoch_name == 'hotstart_epoch')

        # If resuming into main training with LoRA (after hotstart), enable LoRA before loading state dict
        if args.lora_rank > 0 and args.hotstart_training > 0 and not _was_in_hotstart and hasattr(model, 'enable_lora'):
            model.freeze_backbone()
            model.enable_lora(rank=args.lora_rank)
            model.unfreeze_backbone()

        from instanseg.utils.model_loader import remove_module_prefix_from_dict
        _state = remove_module_prefix_from_dict(_preemptable_checkpoint['model_state_dict'])
        model.load_state_dict(_state, strict=True)
        if is_main:
            print(f"[preemptable] Restored model weights, will resume from epoch {_resume_epoch}")

    # Parse --source_dataset, stripping any per-source sampling weight
    # (e.g. [dsb_2018:0.1, open-ai]); weights are kept in args.source_weights and
    # applied by the dataloader sampler (mainly relevant with -w True).
    from instanseg.utils.zarr_loader import parse_source_dataset
    args.source_dataset, args.source_weights = parse_source_dataset(args.source_dataset)
    if is_main:
        print(f"source_dataset={args.source_dataset!r}"
              + (f" source_weights={args.source_weights}" if args.source_weights else ""))


    # Lazy zarr dataset. Built once from the .pth sources on first use (only on
    # the main process; other ranks wait), then read lazily — no in-RAM dataset.
    # Resolution order: explicit --zarr_root > $INSTANSEG_ZARR_ROOT (set by the
    # cluster launcher train.sh, so it only applies on the HPC) > <data_path>/zarr.
    _zarr_env = os.environ.get("INSTANSEG_ZARR_ROOT")
    if args.zarr_root:
        zarr_root = Path(args.zarr_root)
    elif _zarr_env:
        zarr_root = Path(_zarr_env)
    else:
        zarr_root = Path(args.data_path) / "zarr"
    monolith_name = args.dataset if str(args.dataset).endswith(".pth") else f"{args.dataset}_dataset.pth"
    args.manifest_path = ensure_zarr_dataset(zarr_root, data_dir=args.data_path,
                                             monolith_name=monolith_name,
                                             splits=("Train", "Validation"),
                                             accelerator=accelerator,
                                             source_dataset=args.source_dataset)
    if args.find_batch_size:
        from instanseg.utils.AI_utils import find_optimal_batch_size
        from instanseg.utils.zarr_loader import ZarrSegmentationDataset
        model.to(device)
        probe_ds = ZarrSegmentationDataset(args.manifest_path, "Train", args, is_train=True)
        _amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            getattr(accelerator, "mixed_precision", "no"))
        # With drop_last=True a batch larger than the per-epoch/eval sample counts
        # yields an empty loader; cap so train and val always have >=1 batch.
        per_rank = max(1, accelerator.num_processes if accelerator is not None else 1)
        eff_max = min(args.max_batch_size,
                      args.length_of_epoch // per_rank,
                      (args.length_of_eval or args.max_batch_size) // per_rank)
        local_bs = find_optimal_batch_size(model, loss_fn, probe_ds, args, device,
                                           max_bs=max(2, eff_max), mem_fraction=args.bs_mem_fraction,
                                           compiled=bool(args.compile), amp_dtype=_amp,
                                           verbose=is_main)
        if accelerator is not None and accelerator.num_processes > 1:
            # All ranks must agree on batch size; take the min that fits everywhere.
            # NB accelerator.reduce(reduction="min") is NOT honored (falls back to
            # SUM), so use torch.distributed's MIN op directly.
            import torch.distributed as dist
            t = torch.tensor([local_bs], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            local_bs = int(t.item())
        args.batch_size = local_bs
        if is_main:
            print(f"[find_batch_size] using batch_size={args.batch_size}")

    # --- LR scaling by effective (global) batch size --------------------------
    # Done here (batch is final, whether from --find_batch_size or -bs) and before
    # the training-phase optimizers are (re)created. Scales args.lr (and hotstart
    # via _lr_scale_factor) from the (base_batch, lr) reference. The existing
    # 10-epoch warmup + cosine schedule are unchanged.
    n_proc = accelerator.num_processes if accelerator is not None else 1
    eff_batch = args.batch_size * n_proc
    if args.lr_scaling != "none" and args.base_batch_size > 0:
        p = 0.5 if args.lr_scaling == "sqrt" else 1.0
        args._lr_scale_factor = (eff_batch / args.base_batch_size) ** p
    else:
        args._lr_scale_factor = 1.0
    _lr_before = args.lr
    args.lr = args.lr * args._lr_scale_factor
    if is_main:
        print(f"[lr_scaling] {args.lr_scaling}: effective batch {eff_batch} "
              f"(= {args.batch_size} x {n_proc} ranks), base {args.base_batch_size} "
              f"-> lr {_lr_before:.3e} x {args._lr_scale_factor:.3f} = {args.lr:.3e}")

    # DDP sharding is handled by accelerator.prepare(train_loader, test_loader)
    # below — the lazy dataset needs no manual per-rank slicing.
    args._dataloader_was_sharded = False
    train_loader, test_loader = get_zarr_loaders(args)

    if args.show_augmentations and is_main:
        from instanseg.utils.visualization import show_images
        aug_dir = args.output_path / "augmentation_examples"
        os.makedirs(aug_dir, exist_ok=True)
        train_dataset = train_loader.dataset
        for i in range(min(10, len(train_dataset))):
            raw_img, raw_lbl = train_dataset.raw_view(i)
            aug_img, aug_lbl = train_dataset[i][:2]
            show_images(raw_img, raw_lbl, aug_img, aug_lbl,
                        titles=["Raw image", "Raw label", "Aug image", "Aug label"],
                        labels=[1, 3], n_cols=4,
                        save_str=str(aug_dir / f"sample_{i:02d}"))
        print(f"Saved {min(10, len(train_dataset))} augmentation examples to {aug_dir}")

    model.to(device)

    # Re-point method device after Accelerator init (was constructed with cuda:0 default).
    # MAE has no loss module (method is None) — the objective lives in the model wrapper.
    if method is not None:
        method.device = device
        if hasattr(method, 'pixel_classifier') and method.pixel_classifier is not None:
            method.pixel_classifier.to(device)
        if hasattr(method, 'teacher_model') and method.teacher_model is not None:
            method.teacher_model.to(device)
        if hasattr(method, 'loss_temporal') and method.loss_temporal is not None:
            method.loss_temporal.device = device

    model, optimizer = accelerator.prepare(model, optimizer)
    if not args._dataloader_was_sharded:
        train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

    if accelerator.is_main_process:
        print(f"CUDA devices visible: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
        print(f"Accelerator: num_processes={accelerator.num_processes}, mixed_precision={accelerator.mixed_precision}")

    if args.save and is_main:
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        if not os.path.exists(args.output_path / "epoch_outputs"):
            os.mkdir(args.output_path / "epoch_outputs")
        elif not _preemptable_resuming:
            import glob
            for f in glob.glob(str(args.output_path / "epoch_outputs" / "*.png")):
                os.remove(f)
    accelerator.wait_for_everyone()


    if is_main:
        pd.DataFrame.from_dict(args_dict, orient='index').to_csv(args.output_path / "experiment_log.csv",
                                                                header=False)

    iou_threshold = np.linspace(0.5, 1.0, 10)

    # Mixed precision is now driven by Accelerator (mixed_precision="bf16" if fp16),
    # so the manual GradScaler is no longer used — kept None to preserve API.
    scaler = None
    if args.fp16 and is_main:
        print("Mixed precision (bf16) training enabled via Accelerator")

    if args.hotstart_training > 0 and not _preemptable_resuming and not args.mae:
        hot_epochs = args.hotstart_training
        hotstart_lr = 1e-3 * getattr(args, "_lr_scale_factor", 1.0)  # scale with batch like args.lr
        mask_str = f", mask_loss=ce" if args.mask_loss_fn is not None else ""
        if is_main:
            print(f"Hotstart for {hot_epochs} epochs with seed_loss=l1_distance, instance_loss=dice_loss{mask_str}, lr={hotstart_lr}")

        method.update_seed_loss("l1_distance")
        method.update_instance_loss("dice_loss")
        if args.mask_loss_fn is not None:
            method.update_mask_loss("ce")

        # Freeze pretrained backbone weights during hotstart
        _backbone_frozen = hasattr(model, 'freeze_backbone')
        if _backbone_frozen:
            if is_main:
                print("Freezing backbone weights for hotstart")
            model.freeze_backbone()
            optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args, lr=hotstart_lr)
        else:
            optimizer = get_optimizer(model.parameters(), args, lr=hotstart_lr)

        # --warmup_phase hotstart: ramp the LR DURING hotstart (up to hotstart_lr,
        # over min(warmup_epochs, hot_epochs)) instead of in main. Built here so it
        # is attached to the hotstart optimizer that main() actually steps.
        if args.warmup_phase == "hotstart" and args.warmup_epochs > 0:
            from torch.optim.lr_scheduler import LambdaLR
            _we = min(args.warmup_epochs, hot_epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / _we))
            if is_main:
                print(f"[warmup] ramping LR over the first {_we} hotstart epochs (up to {hotstart_lr:.3e})")
        else:
            scheduler = None  # hotstart runs flat at hotstart_lr (warmup, if any, is in main)

        model, train_losses, test_losses, f1_list, f1_list_cells = main(model, loss_fn, train_loader, test_loader, num_epochs=hot_epochs, epoch_name='hotstart_epoch', scaler=scaler)

        # Unfreeze backbone weights after hotstart
        if _backbone_frozen:
            if args.lora_rank > 0 and hasattr(model, 'enable_lora'):
                if is_main:
                    print(f"Enabling LoRA (rank={args.lora_rank}) for main training")
                model.enable_lora(rank=args.lora_rank)
                model.unfreeze_backbone()
            else:
                if is_main:
                    print("Unfreezing backbone weights for main training")
                model.unfreeze_backbone()

        optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)

        # Recreate scheduler for the new optimizer. Suppress main-phase warmup when
        # it was placed in the hotstart phase instead (--warmup_phase hotstart).
        _main_warmup = 0 if (args.warmup_phase == "hotstart" and args.hotstart_training > 0) else None
        scheduler = _build_scheduler(optimizer, args, warmup=_main_warmup)

        mask_str = f", mask_loss={args.mask_loss_fn}" if args.mask_loss_fn is not None else ""
        if is_main:
            print(f"Starting main training with seed_loss={args.seed_loss_fn}, instance_loss={args.instance_loss_fn}{mask_str}, lr={args.lr}")
        method.update_seed_loss(args.seed_loss_fn)
        method.update_instance_loss(args.instance_loss_fn)
        if args.mask_loss_fn is not None:
            method.update_mask_loss(args.mask_loss_fn)
        method.reset_uncertainty_weights()

        # Leiden cluster-weighted resampling depended on the in-RAM dataset +
        # the old get_loaders; not yet ported to the lazy zarr pipeline.
        if args.cluster_sample:
            raise NotImplementedError(
                "--cluster_sample (Leiden cluster-weighted sampling) is not yet "
                "supported by the zarr data pipeline. Use --weight for "
                "parent-dataset frequency balancing, or set --cluster_sample 0.")

    _resample_kwargs = {}

    if _preemptable_resuming and _preemptable_checkpoint is not None:
        _ckpt = _preemptable_checkpoint
        _prior_train = _ckpt.get('train_losses', [])
        _prior_test = _ckpt.get('test_losses', [])
        _prior_f1 = _ckpt.get('f1_list', [])
        _prior_f1_cells = _ckpt.get('f1_list_cells', [])
        _epoch_name = _ckpt.get('epoch_name', 'output_epoch')
        _prior_best_f1 = _ckpt.get('f1_score', -1)
        _start = _resume_epoch

        # Determine if we were in hotstart or main training phase using the saved epoch_name
        if args.hotstart_training > 0 and _was_in_hotstart:
            # Resume into hotstart phase: replicate hotstart setup
            if is_main:
                print(f"[preemptable] Resuming hotstart phase from epoch {_start}")
            method.update_seed_loss("l1_distance")
            method.update_instance_loss("dice_loss")
            if args.mask_loss_fn is not None:
                method.update_mask_loss("ce")

            # Freeze backbone like the original hotstart does
            _backbone_frozen = hasattr(model, 'freeze_backbone')
            if _backbone_frozen:
                model.freeze_backbone()
                optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args, lr=1e-3)
            else:
                optimizer = get_optimizer(model.parameters(), args, lr=1e-3)
            optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
            # Scheduler is None during hotstart
            scheduler = None

            model, train_losses, test_losses, f1_list, f1_list_cells = main(
                model, loss_fn, train_loader, test_loader,
                num_epochs=args.hotstart_training, epoch_name='hotstart_epoch',
                prior_train_losses=_prior_train, prior_test_losses=_prior_test,
                prior_f1_list=_prior_f1, prior_f1_list_cells=_prior_f1_cells,
                scaler=scaler, start_epoch=_start, prior_best_f1=_prior_best_f1)

            # Continue to main training after hotstart completes: unfreeze backbone
            if _backbone_frozen:
                if args.lora_rank > 0 and hasattr(model, 'enable_lora'):
                    model.enable_lora(rank=args.lora_rank)
                    model.unfreeze_backbone()
                else:
                    model.unfreeze_backbone()

            optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
            _main_warmup = 0 if (args.warmup_phase == "hotstart" and args.hotstart_training > 0) else None
            scheduler = _build_scheduler(optimizer, args, warmup=_main_warmup)

            method.update_seed_loss(args.seed_loss_fn)
            method.update_instance_loss(args.instance_loss_fn)
            if args.mask_loss_fn is not None:
                method.update_mask_loss(args.mask_loss_fn)
            method.reset_uncertainty_weights()

            model, train_losses, test_losses, f1_list, f1_list_cells = main(
                model, loss_fn, train_loader, test_loader, num_epochs=num_epochs,
                prior_train_losses=train_losses, prior_test_losses=test_losses,
                prior_f1_list=f1_list, prior_f1_list_cells=f1_list_cells,
                hotstart_epoch=args.hotstart_training, scaler=scaler,
                **_resample_kwargs)
        else:
            # Resume into main training phase: recreate optimizer with correct param groups, then restore state
            if args.channel_invariant and args.freeze_main_model:
                _inner = accelerator.unwrap_model(model)
                optimizer = get_optimizer(_inner.model.AdaptorNet.parameters(), args)
            else:
                optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
            optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in _ckpt:
                scheduler.load_state_dict(_ckpt['scheduler_state_dict'])

            model, train_losses, test_losses, f1_list, f1_list_cells = main(
                model, loss_fn, train_loader, test_loader, num_epochs=num_epochs,
                epoch_name=_epoch_name,
                prior_train_losses=_prior_train, prior_test_losses=_prior_test,
                prior_f1_list=_prior_f1, prior_f1_list_cells=_prior_f1_cells,
                hotstart_epoch=args.hotstart_training if args.hotstart_training > 0 else None,
                scaler=scaler, start_epoch=_start, prior_best_f1=_prior_best_f1,
                **_resample_kwargs)

        del _preemptable_checkpoint
    elif args.hotstart_training > 0:
        model, train_losses, test_losses, f1_list, f1_list_cells = main(model, loss_fn, train_loader, test_loader, num_epochs=num_epochs,
            prior_train_losses=train_losses, prior_test_losses=test_losses,
            prior_f1_list=f1_list, prior_f1_list_cells=f1_list_cells,
            hotstart_epoch=args.hotstart_training, scaler=scaler,
            **_resample_kwargs)
    else:
        model, train_losses, test_losses, f1_list, f1_list_cells = main(model, loss_fn, train_loader, test_loader, num_epochs=num_epochs, scaler=scaler,
            **_resample_kwargs)

    accelerator.wait_for_everyone()
    if not is_main:
        return

    if not args.mae:
        # Sanity-reload the exported segmentation checkpoint. Skipped for MAE:
        # the saved weights are the MAEWrapper (backbone.* keys) and there is no
        # segmentation export path for the reconstruction model yet.
        from instanseg.utils.model_loader import load_model
        model, model_dict = load_model(folder="", path=args.output_path) #Load model from checkpoint
        model.eval()
        model.to(device)

    df = pd.DataFrame({"train_loss": train_losses, "test_loss": test_losses, "f1_score": f1_list})
    df.to_csv(args.output_path / "experiment_metrics.csv", index=False, header=True)


    fig = plot_average(train_losses, test_losses, window_size=len(train_losses) // 10 + 1)
    plt.savefig(args.output_path / "loss.png")
    plt.close()

    if args.dual_head_output:
        fig = plt.plot(f1_list, label="f1 score nuclei")
        plt.plot(f1_list_cells, label="f1 score cells")
        plt.ylim(0, 1)
        plt.savefig(args.output_path / "f1_metric.png")
        plt.legend()
        plt.close()

    else:
        fig = plt.plot(f1_list, label="f1 score")
        plt.ylim(0, 1)
        plt.savefig(args.output_path / "f1_metric.png")
        plt.close()

    if not args.on_cluster and args.experiment_str is None:
        experiment_str = "experiment"
    elif args.experiment_str is not None:
        experiment_str = args.experiment_str



if __name__ == "__main__":
    instanseg_training()