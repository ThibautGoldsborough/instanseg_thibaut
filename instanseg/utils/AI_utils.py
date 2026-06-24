import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from instanseg.utils.metrics import _robust_average_precision, _robust_f1_mean_calculator

from instanseg.utils.augmentations import Augmentations
import time

from instanseg.utils.utils import show_images
import warnings


def _channel_classes(target_segmentation: str) -> list[str]:
    """Map label-channel index -> AdaLN condition name.

    data_loader._format_labels stacks (nucleus, cell) for the two-target case,
    so channel 0 is always "N" and channel 1 is "C". For a single target the
    only channel is whatever letter was requested ("N" or "C").
    """
    return ["N", "C"] if len(target_segmentation) == 2 else [target_segmentation]


def _pick_condition(labels: torch.Tensor,
                    class_for_channel: list[str],
                    generator: "torch.Generator | None" = None):
    """Reduce a (B, K, H, W) label tensor (K in {1,2}, absent channel == all -1)
    to a single label map per sample plus its AdaLN condition.

    For each sample we pick one non-empty channel uniformly at random. Returns
    ``(single_labels (B,1,H,W), conditions)`` where ``conditions`` is a device
    LongTensor of chosen channel indices — these equal the AdaLN class indices
    (both ordered ("N","C"): channel 0 = nucleus = "N", channel 1 = cell = "C"),
    so the model consumes them directly via ``AdaLNConditioner.encode`` with no
    host sync. Samples with no instance-bearing channel fall back to channel 0;
    the loss treats those as empty anyway. ``class_for_channel`` is retained for
    that index↔class correspondence (the model maps the indices to embeddings).

    Fully on-device: no ``.cpu()`` / Python loop, so it adds no per-step GPU→CPU
    sync (those stalls were hurting GPU utilisation, especially once the backbone
    is compiled).

    "Non-empty" == contains at least one positive instance id (``> 0``). We test
    ``<= 0`` rather than ``< 0`` so the absence check is robust to padding: an
    absent map is the all-(-1) sentinel, but augmentation/padding can fill it
    with 0 (background), making it a -1/0 mix that ``< 0`` would mis-read as
    present. Background (0) carries no instance to segment, so a 0/-1 map is
    correctly "empty" under ``<= 0``.
    """
    B, K = labels.shape[0], labels.shape[1]
    # non_empty[b, c]: channel c of sample b has any instance. All on-device.
    non_empty = ~(labels <= 0).all(dim=3).all(dim=2)  # (B, K) bool
    # Uniform pick among non-empty channels == argmax of iid uniforms with empty
    # channels masked to -1 (the max of iid uniforms is a uniform index). All
    # channels empty -> argmax picks channel 0.
    if generator is not None:
        scores = torch.rand(B, K, generator=generator).to(labels.device)
    else:
        scores = torch.rand(B, K, device=labels.device)
    chosen = scores.masked_fill(~non_empty, -1.0).argmax(dim=1)  # (B,), on device
    single_labels = labels.gather(
        1, chosen.view(B, 1, 1, 1).expand(B, 1, labels.shape[2], labels.shape[3]))
    return single_labels, chosen


def find_optimal_batch_size(model, loss_fn, dataset, args, device,
                            max_bs: int = 512, mem_fraction: float = 0.9,
                            probe_sizes: tuple[int, int] = (2, 4),
                            pool_size: int = 32, compiled: bool = False,
                            amp_dtype=None, verbose: bool = True) -> int:
    """Estimate the largest batch size that fits in GPU memory and return it.

    Runs real forward+backward steps (activation+gradient memory captured) but NOT
    ``optimizer.step()`` — weights are never mutated, so no warm-start corruption
    and no state restore. AdamW optimizer state is reserved analytically.

    ``compiled``: match the training mode. When the model is ``torch.compile``d,
    measure compiled memory (accurate) by calling ``torch._dynamo.reset()`` before
    each probe size — every size compiles from a clean dynamo state, which avoids
    the cross-size guard contamination that otherwise trips an inductor recompile
    assert; a final ``reset()`` leaves a clean slate so training compiles fresh.
    Costs a full recompile per probe size (a few minutes), negligible before a long
    run. When not compiled, the model is eager already and no reset is needed.

    Measures *steady-state* peak (resets stats after a warmup step), then linearly
    extrapolates peak(bs) = fixed + per_sample·bs to the memory budget — only 2
    measurements + 1 verify, no deliberate OOM. Rounds down to a multiple of 8 and
    backs off 15% if the verify step doesn't fit.
    """
    if not torch.cuda.is_available() or len(dataset) == 0:
        return getattr(args, "batch_size", probe_sizes[0])

    # The maxvit+AdaLN compiled graph hits an inductor size/stride assert at very
    # small batch sizes (bs 2/4 crash; bs>=8 compile fine), so probe larger sizes
    # when compiled. The verify step backs off gracefully if a size won't compile.
    if compiled and probe_sizes == (2, 4):
        probe_sizes = (8, 16)

    use_adaln = getattr(args, "adaln", False)
    class_for_channel = _channel_classes(args.target_segmentation) if use_adaln else None
    total_mem = torch.cuda.get_device_properties(device).total_memory
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt_reserve = 2 * n_params * 4  # AdamW: 2 fp32 moments/param (probe skips optim.step)
    budget = mem_fraction * total_mem - opt_reserve

    # Bias the probe pool toward the DENSEST images (most instances => most
    # instance-loss crops + retained autograd graph). Loss memory scales with
    # per-batch instance content, so probing average-density items underestimates
    # worst-case batches and OOMs mid-epoch. Estimate instances/image from the
    # manifest as H*W / median_instance_area; fall back to uniform random.
    rng = np.random.default_rng(0)
    n = len(dataset)
    rows = getattr(dataset, "rows", None)
    pool_idx = None
    if rows is not None and {"median_nucleus_area", "median_cell_area",
                             "height", "width"}.issubset(rows.columns):
        area = rows[["median_nucleus_area", "median_cell_area"]].min(axis=1)
        area = area.fillna(area.max()).clip(lower=1.0)
        density = (rows["height"] * rows["width"]) / area  # ~instances per image
        pool_idx = np.argsort(density.to_numpy())[::-1][:min(pool_size, n)]
    if pool_idx is None:
        pool_idx = rng.integers(0, n, size=min(pool_size, n))
    pool = [dataset[int(i)] for i in pool_idx]

    def _make_batch(bs):
        items = [pool[j % len(pool)] for j in range(bs)]
        imgs, labs, _ = collate_fn([(it[0], it[1]) for it in items])
        return imgs.to(device), labs.to(device)

    def _reset_dynamo():
        if compiled and hasattr(torch, "_dynamo"):
            torch._dynamo.reset()  # fresh compile per size; clears contaminating guards

    model.train()

    import contextlib
    def _autocast():
        return (torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None else contextlib.nullcontext())

    def _step(images, labels):
        model.zero_grad(set_to_none=True)
        with _autocast():  # match training's mixed precision (memory + compile context)
            if use_adaln:
                lab, cond = _pick_condition(labels, class_for_channel)
                out = model(images, condition=cond)
            else:
                lab, out = labels, model(images)
            loss = loss_fn(out, lab.clone()).mean()
        loss.backward()

    def _measure(bs):
        """Steady-state peak bytes for a fwd+bwd at batch size ``bs`` (warmup +
        measured step). Returns None on OOM or, when compiled, on a compile failure
        at this size (so the search backs off instead of crashing training)."""
        try:
            _reset_dynamo()                             # fresh compile per size (compiled only)
            torch.cuda.empty_cache()
            images, labels = _make_batch(bs)
            _step(images, labels)                       # warmup (compiles if compiled)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            _step(images, labels)                       # measured
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            del images, labels
            return peak
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "out of memory" in msg:
                torch.cuda.empty_cache()
                return None
            if compiled:  # a compile failure at this size — skip it, don't crash startup
                if verbose:
                    print(f"[find_batch_size] bs={bs}: compile/probe failed ({type(e).__name__}: "
                          f"{str(e)[:120]}) — treating as does-not-fit")
                torch.cuda.empty_cache()
                return None
            raise

    mode = "compiled" if compiled else "eager"
    try:
        b1, b2 = probe_sizes
        m1, m2 = _measure(b1), _measure(b2)
        if m1 is None or m2 is None or m2 <= m1:
            chosen = b1 if m1 is not None else max(1, b1 // 2)
            if verbose:
                print(f"[find_batch_size] probe OOM/degenerate at small sizes; using {chosen}")
            return chosen
        per_sample = (m2 - m1) / (b2 - b1)
        fixed = m1 - per_sample * b1
        est = int((budget - fixed) / per_sample)
        est = max(b2, min(est, max_bs))
        chosen = max(b2, (est // 8) * 8)               # round down to multiple of 8

        peak = _measure(chosen)                         # verify
        while (peak is None or peak > budget) and chosen > b2:
            chosen = max(b2, int(chosen * 0.85) // 8 * 8)
            peak = _measure(chosen)

        if verbose:
            verified = f"{peak/1e9:.2f}GB" if peak is not None else "n/a"
            print(f"[find_batch_size] {mode} probe | GPU={total_mem/1e9:.1f}GB "
                  f"budget={budget/1e9:.1f}GB ({int(mem_fraction*100)}% - {opt_reserve/1e9:.1f}GB optim) | "
                  f"peak(bs={b1})={m1/1e9:.2f}GB peak(bs={b2})={m2/1e9:.2f}GB "
                  f"~{per_sample/1e9:.3f}GB/sample | estimate {est}, verified bs={chosen} @ {verified}")
            note = "" if compiled else " (eager probe; with --compile the true fit is usually larger)"
            print(f"[find_batch_size] selected batch_size = {chosen}{note}")
        return chosen
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        _reset_dynamo()  # clean dynamo slate so training compiles fresh at the chosen size


global_step = 0
def train_epoch(train_model,
                train_device,
                train_dataloader,
                train_loss_fn,
                train_optimizer,
                args,
                scaler=None,
                ):

    global global_step
    start = time.time()
    train_model.train()
    train_loss = []
    skip_bad_batches = getattr(args, 'skip_bad_batches', True)
    skipped_loss = 0
    skipped_grad = 0

    accelerator = getattr(args, "accelerator", None)
    is_main = accelerator.is_main_process if accelerator is not None else True
    device = accelerator.device if accelerator is not None else train_device

    use_adaln = getattr(args, "adaln", False)
    class_for_channel = _channel_classes(args.target_segmentation) if use_adaln else None

    # Optional per-epoch timing breakdown (data-load wait vs GPU compute) to
    # diagnose dataloader bottlenecks (low GPU util). Adds per-iter cuda syncs,
    # so it slows the run slightly and is off by default.
    prof = getattr(args, "time_dataloading", False)
    use_cuda = torch.cuda.is_available()

    def _sync():
        if prof and use_cuda:
            torch.cuda.synchronize(device)

    t_data = t_h2d = t_cond = t_model = t_loss = t_bwd = t_step = 0.0
    n_iter = 0
    _end = time.time()

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster or not is_main):
        if prof:
            _t_iter = time.time()
            t_data += _t_iter - _end  # time blocked waiting on the dataloader

        image_batch = image_batch.to(device, non_blocking=True)
        labels = labels_batch.to(device, non_blocking=True)
        if prof:
            _sync(); _m_h2d = time.time(); t_h2d += _m_h2d - _t_iter

        _mark = _m_h2d if prof else None
        if use_adaln:
            # Pick one label map (C or N) per sample and tell the model which.
            labels, conditions = _pick_condition(labels, class_for_channel)
            if prof:
                _sync(); _t = time.time(); t_cond += _t - _mark; _mark = _t
            output = train_model(image_batch, condition=conditions)
        else:
            output = train_model(image_batch)
        if prof:
            _sync(); _t = time.time(); t_model += _t - _mark; _mark = _t
        loss = train_loss_fn(output, labels.clone()).mean()
        if prof:
            _sync(); _m_loss = time.time(); t_loss += _m_loss - _mark

        # DDP: sync the skip decision across ranks. If any rank has a non-finite
        # loss, *all* ranks must skip — otherwise the surviving ranks call
        # accelerator.backward(...) and hang in NCCL waiting for the rank that
        # bailed out via `continue`. Note: `not torch.isfinite(loss)` is a
        # Python bool (0-d tensor → __bool__), so use `~` to keep it as a tensor.
        local_bad_loss = (~torch.isfinite(loss)).to(torch.float32).detach()
        if accelerator is not None and accelerator.num_processes > 1:
            any_bad_loss = accelerator.reduce(local_bad_loss, reduction="sum").item() > 0
        else:
            any_bad_loss = local_bad_loss.item() > 0

        if skip_bad_batches and any_bad_loss:
            if local_bad_loss.item() > 0:
                warnings.warn(f"Skipping batch: non-finite loss ({loss.item()})")
            train_optimizer.zero_grad(set_to_none=True)
            skipped_loss += 1
            if prof:
                _end = time.time()
            continue

        train_optimizer.zero_grad()
        if accelerator is not None:
            accelerator.backward(loss)
            total_norm = accelerator.clip_grad_norm_(train_model.parameters(), args.clip)
        else:
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)
        if prof:
            # includes the non-finite-loss check above (a small .item() host sync)
            _sync(); _m_bwd = time.time(); t_bwd += _m_bwd - _m_loss

        # accelerator.clip_grad_norm_ returns the synced (global) norm, so the
        # skip decision below is already identical on every rank — no extra
        # all-reduce needed here.
        if skip_bad_batches and not torch.isfinite(total_norm):
            warnings.warn(f"Skipping batch: non-finite gradient norm ({total_norm.item()})")
            train_optimizer.zero_grad(set_to_none=True)
            skipped_grad += 1
            if prof:
                _end = time.time()
            continue
        train_optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())
        if prof:
            _sync(); _t = time.time(); t_step += _t - _m_bwd
            n_iter += 1
            _end = time.time()

    end = time.time()

    if prof and is_main:
        wall = end - start
        compute = t_h2d + t_cond + t_model + t_loss + t_bwd + t_step
        imgs = n_iter * args.batch_size
        ni = max(n_iter, 1)
        print(f"[timing] wall={wall:.1f}s | dataload_wait={t_data:.1f}s ({100*t_data/wall:.0f}%) "
              f"compute={compute:.1f}s ({100*compute/wall:.0f}%) | "
              f"{n_iter} iters, {imgs/wall:.1f} img/s"
              + (f" (x{accelerator.num_processes} ranks ~= {imgs/wall*accelerator.num_processes:.1f} global)"
                 if accelerator is not None and accelerator.num_processes > 1 else ""))
        print(f"[timing]   per-iter ms: h2d={1000*t_h2d/ni:.1f} adaln_pick={1000*t_cond/ni:.1f} "
              f"model_fwd={1000*t_model/ni:.1f} loss={1000*t_loss/ni:.1f} "
              f"backward={1000*t_bwd/ni:.1f} optim_step={1000*t_step/ni:.1f}")

    if (skipped_loss or skipped_grad) and is_main:
        print(f"[skip_bad_batches] skipped {skipped_loss} batch(es) for non-finite loss, "
              f"{skipped_grad} for non-finite gradient norm")

    mean_loss = float(np.mean(train_loss)) if train_loss else float('nan')
    return mean_loss, end - start
    

global_step_test = 0
def test_epoch(test_model,
               test_device,
               test_dataloader,
               test_loss_fn,
               args,
               postprocessing_fn,
               method,
               iou_threshold,
               debug=False,
               save_str=None,
               save_bool=False,
               best_f1=None,
               use_amp=False):
    global global_step_test
    start = time.time()

    accelerator = getattr(args, 'accelerator', None)
    is_main = accelerator.is_main_process if accelerator is not None else True
    device = accelerator.device if accelerator is not None else test_device

    test_model.eval()
    test_loss = []

    use_adaln = getattr(args, "adaln", False)
    use_mae = getattr(args, "mae", False)
    class_for_channel = _channel_classes(args.target_segmentation) if use_adaln else None
    # Seed the per-sample C/N pick so validation conditions are stable across
    # epochs (comparable F1) rather than reshuffling every evaluation.
    cond_generator = torch.Generator().manual_seed(0) if use_adaln else None

    current_f1_list = []
    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster or not is_main):
            image_batch = image_batch.to(device, non_blocking=True)
            labels = labels_batch.to(device, non_blocking=True)
            if use_adaln:
                labels, conditions = _pick_condition(labels, class_for_channel, generator=cond_generator)
                output = test_model(image_batch, condition=conditions)
            else:
                output = test_model(image_batch)
            loss = test_loss_fn(output, labels.clone()).mean()
            test_loss.append(loss.detach().cpu().numpy())

            if use_mae:
                # Reconstruction objective: no instances to postprocess / score.
                global_step_test += 1
                continue

            if type(output) == list:
                output = output[0]

            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':
                predicted_labels = torch.stack([postprocessing_fn(out) for out in output])
                f1i = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                               threshold=iou_threshold)

                current_f1_list.append((f1i))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)

            global_step_test += 1

    # Empty val loader (no batches iterated): bail out gracefully instead of an
    # UnboundLocalError on image_batch/test_loss below. Consistent across ranks
    # (even_batches keeps per-rank counts equal, so all ranks see 0), so no
    # collective is entered unevenly. Should be rare now that the eval loader uses
    # drop_last=False; left as a guard against a too-small --length_of_eval.
    if len(test_loss) == 0:
        if is_main:
            warnings.warn("Validation loader produced no batches — skipping evaluation "
                          "this epoch (check --length_of_eval vs batch_size / num GPUs).")
        n_cols = 2 if getattr(args, "dual_head_output", False) else 1
        return float('nan'), np.full(n_cols, np.nan), time.time() - start

    if use_mae:
        # MAE: report reconstruction loss only; F1 / overlays don't apply. Save
        # diagnostic panels (original / masked / reconstruction / composite) for
        # the first epochs or whenever the recon loss improves (best_f1 carries
        # the prior best -loss, mirroring the segmentation save logic).
        recon_loss = float(np.mean(test_loss))
        if (save_bool or (best_f1 is not None and -recon_loss > best_f1)) and is_main:
            from instanseg.utils.loss.mae import mae_make_panels
            images, titles = mae_make_panels(output,
                                             patch_size=getattr(args, "mae_patch_size", 16),
                                             norm_target=getattr(args, "mae_norm_target", True))
            show_images(images, save_str=save_str, titles=titles, n_cols=4, colorbar=False)
        return recon_loss, np.array([np.nan], dtype=np.float32), time.time() - start

    # Gather per-batch F1 values across ranks so mean_f1 reflects the whole val set.
    if accelerator is not None and accelerator.num_processes > 1 and len(current_f1_list) > 0:
        f1_local = np.stack([np.atleast_1d(x) for x in current_f1_list]).astype(np.float32)
        f1_t = torch.tensor(f1_local, device=accelerator.device)
        f1_t = accelerator.gather_for_metrics(f1_t).detach().cpu().numpy()
        current_f1_list = list(f1_t)

    f1_array = np.array(current_f1_list)  # either N,2 or N,

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)

    if (mean_f1 > best_f1 or save_bool) and is_main:
        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]
        labels_dst = labels[0]
        lab = postprocessing_fn(output[0])

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()
    return np.mean(test_loss), mean1_f1, end - start


def collate_fn(data):
    # data is of length batch size
    # data[0][0] is first image, data[0][1] os the first label

    # print(data[0][0].shape,len(data))
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    imgs, labels = zip(*data)
    lengths = [img.shape[0] for img in imgs]

    max_len = max(lengths)
    C, H, W = data[0][0].shape
    images = torch.zeros((len(data), max_len, H, W))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    for i, img in enumerate(imgs):
        images[i, :len(img)] = img

    return images, labels, lengths.int()


# import fastremap
class Segmentation_Dataset():
    def __init__(self, img, 
                label, 
                common_transforms=True,
                metadata=None, 
                size=(256, 256), 
                augmentation_dict=None,
                dim_in=3, 
                debug=False, 
                cells_and_nuclei=False, 
                target_segmentation="N", 
                channel_invariant = False,
                random_seed = None):
        
        self.X = img
        self.Y = label
        self.common_transforms = common_transforms

        assert len(self.X) == len(self.Y), "The number of images and labels must be the same"
        if len(metadata) == 0:
            self.metadata = [None] * len(self.X)
        else:
            self.metadata = metadata

        assert len(self.X) == len(self.metadata), print("The number of images and metadata must be the same")
        self.size = size
        self.Augmenter = Augmentations(augmentation_dict=augmentation_dict, 
                                       debug=debug, 
                                       shape=self.size,
                                       dim_in=dim_in, 
                                       cells_and_nuclei=cells_and_nuclei,
                                       target_segmentation=target_segmentation, 
                                       channel_invariant = channel_invariant,
                                       random_seed = random_seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        data = self.X[i]
        label = self.Y[i]
        meta = self.metadata[i]

        if self.common_transforms:
            data, label = self.Augmenter(data, label, meta)

        if len(label.shape) == 2:
            label = label[None, :]
        if len(data.shape) == 2:
            data = data[None, :]

        assert not data.isnan().any(), "Tranformed images contains NaN"
        assert not label.isnan().any(), "Transformed labels contains NaN"

        return data.float(), label



def plot_loss(_model):
    loss_fig = plt.figure()
    timer = loss_fig.canvas.new_timer(interval=300000)
    timer.add_callback(plt.close)

    losses = [param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None]
    names = [name for name, param in _model.named_parameters() if param.grad is not None]

    plt.plot(losses)
    plt.xticks(np.arange(len(names))[::1], names[::1])
    plt.xticks(fontsize=8, rotation=90)
    spacing = 0.5
    loss_fig.subplots_adjust(bottom=spacing)
    timer.start()
    plt.show()



def check_max_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.max()


def check_min_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.min()


def check_mean_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.mean()

def optimize_hyperparameters(model,postprocessing_fn,
                              data_loader = None,
                              val_images = None,
                              val_labels = None,
                              max_evals = 50,
                              verbose = False,
                              threshold = [0.5, 0.7, 0.9],
                              show_progressbar = True,
                              device = None,
                              exclude_params = None):


    from instanseg.utils.metrics import _robust_average_precision
    from instanseg.utils.utils import _choose_device

    from hyperopt import fmin
    from hyperopt import hp
    from hyperopt import Trials
    from hyperopt import tpe
    import copy

    if device is None:
        device = _choose_device()

    bayes_trials = Trials()

    space = {  # instanseg
        'mask_threshold': hp.uniform('mask_threshold', 0.3, 0.7),
        'seed_threshold': hp.uniform('seed_threshold', 0.3, 0.9),
        'fg_threshold': hp.uniform('fg_threshold', 0.3, 0.7),
        'overlap_threshold': hp.uniform('overlap_threshold', 0.1, 0.9),
        #'min_size': hp.uniform('min_size', 0, 30),
        'peak_distance': hp.uniform('peak_distance', 3, 10),
      #  'mean_threshold': hp.uniform('mean_threshold', 0.0, 0.5)} #the max could be increased, but may cause the method not to converge for some reason.
        }
    if exclude_params:
        for p in exclude_params:
            space.pop(p, None)
    _model = model # copy.deepcopy(model)
    _model.eval()
    predictions = []

    with torch.no_grad():
        if data_loader is not None:
            for image_batch, labels_batch, _ in data_loader:
                    image_batch = image_batch.to(device)
                    output = _model(image_batch).cpu()
                    predictions.extend([pred,masks] for pred,masks in zip(output,labels_batch))


            def objective(params={}):
                pred_masks = []
                gt_masks = []
                for pred, masks in predictions:
                    lab = postprocessing_fn(pred.to(device), **params).cpu()
                    pred_masks.append(lab)
                    gt_masks.append(masks)

                mean_f1 = _robust_average_precision(torch.stack(gt_masks),torch.stack(pred_masks),threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        
        elif val_images is not None and val_labels is not None:
            from instanseg.utils.tiling import _instanseg_padding, _recover_padding
            def objective(params={}):
                pred_masks = []
                gt_masks = []
                #randomly shuffle val_images and val_labels

                np.random.seed(0)
                indexes = np.random.permutation(len(val_images))[:300]
                indexes.sort()

                for i in indexes:
                    imgs = val_images[i]
                    gt_mask = val_labels[i]
                    with torch.no_grad():
                        imgs = imgs.to(device)
                        imgs, pad = _instanseg_padding(imgs, min_dim = 32)
                        output = _model(imgs[None,])
                        output = _recover_padding(output, pad).squeeze(0)
                        lab = postprocessing_fn(output.to(device), **params).cpu()
                        pred_masks.append(lab)
                        gt_masks.append(gt_mask)

                mean_f1 = _robust_average_precision(gt_masks,pred_masks,threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        else:
            raise ValueError("Either data_loader or val_images and val_labels must be provided")

        print("Optimizing hyperparameters")
        # Optimize
        best = fmin(fn=objective, 
                    space=space, 
                    algo=tpe.suggest,
                    max_evals=max_evals, 
                    trials=bayes_trials, 
                    rstate=np.random.default_rng(0),
                    show_progressbar = show_progressbar)
    
    if verbose:
        print(best)
    return best



