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
    sup_loss_hist = []
    cons_loss_hist = []
    use_amp = scaler is not None
    skip_bad_batches = getattr(args, 'skip_bad_batches', True)
    skipped_loss = 0
    skipped_grad = 0

    pct_unsup = float(getattr(args, 'percent_unsupervised', 0.0) or 0.0)
    use_unsup = pct_unsup > 0
    if use_unsup:
        from instanseg.utils.loss.consistency_loss import consistency_loss
        dim_coords = int(getattr(args, 'dim_coords', 2))
        cons_weight = float(getattr(args, 'consistency_weight', 1.0))
        cons_nan_dumped = False  # one-shot diagnostic dump
        sup_nan_dumped = False

        def _offset_only(x: torch.Tensor) -> torch.Tensor:
            # Match InstanSeg's displacement formula (instanseg_loss.py:1354):
            # spatial_emb = (sigmoid(raw) - 0.5) * 8 + xxyy. The consistency loss
            # operates on the displacement (offset from xxyy), in coord units.
            #
            # Channel-order swap: InstanSeg's xxyy = cat((xx, yy)) has channel 0=x
            # and channel 1=y, but consistency_loss treats channel 0=y, channel 1=x
            # (its homography solver is _compute_homography_yx). Flip channels so
            # the displacement vector enters consistency_loss in (y, x) order.
            out = train_model(x)
            if isinstance(out, list):
                out = out[0]
            raw = out[:, :dim_coords]
            disp_xy = (torch.sigmoid(raw) - 0.5) * 8.0
            return disp_xy[:, [1, 0]]

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):

        image_batch = image_batch.to(train_device)
        labels = labels_batch.to(train_device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if use_unsup:
                # Wiped samples are stamped with sentinel value -1 (chosen to match
                # the augmenter's "skip renumbering" convention). Augmentation may pad
                # borders with 0, so detect via .any() rather than .all().
                unsup_mask = (labels == -1).reshape(labels.shape[0], -1).any(dim=1)
                sup_mask = ~unsup_mask

                loss_terms = []
                if sup_mask.any():
                    sup_imgs = image_batch[sup_mask]
                    sup_labels = labels[sup_mask]
                    sup_output = train_model(sup_imgs)
                    sup_loss = train_loss_fn(sup_output, sup_labels.clone()).mean()
                    if torch.isfinite(sup_loss):
                        loss_terms.append(sup_loss)
                        sup_loss_hist.append(sup_loss.detach().cpu().item())
                    else:
                        if not sup_nan_dumped:
                            sup_nan_dumped = True
                            stats = (
                                f"sup_imgs: shape={tuple(sup_imgs.shape)} "
                                f"min={sup_imgs.min().item():.4g} max={sup_imgs.max().item():.4g} "
                                f"finite={torch.isfinite(sup_imgs).all().item()}; "
                                f"sup_labels: shape={tuple(sup_labels.shape)} "
                                f"unique~={sup_labels.unique()[:8].tolist()} "
                                f"finite={torch.isfinite(sup_labels.float()).all().item()}; "
                                f"sup_output: min={sup_output.min().item():.4g} max={sup_output.max().item():.4g} "
                                f"finite={torch.isfinite(sup_output).all().item()}; "
                                f"sup_loss={sup_loss.item()}"
                            )
                            try:
                                import os
                                dump_dir = getattr(args, 'output_path', None)
                                if dump_dir is not None:
                                    dump_path = os.fspath(dump_dir) + "/supervised_nan_dump.pt"
                                    torch.save({
                                        'sup_imgs': sup_imgs.detach().cpu(),
                                        'sup_labels': sup_labels.detach().cpu(),
                                        'sup_output': sup_output.detach().cpu(),
                                        'sup_loss': sup_loss.detach().cpu(),
                                    }, dump_path)
                                    stats += f"  saved to {dump_path}"
                            except Exception as _e:
                                stats += f"  (dump failed: {_e})"
                            warnings.warn(f"sup_loss is non-finite; dropping. {stats}")
                        else:
                            warnings.warn(f"sup_loss non-finite ({sup_loss.item()}); dropping (suppressed further detail)")
                if unsup_mask.any() and cons_weight != 0:
                    unsup_imgs = image_batch[unsup_mask]
                    cons_loss = consistency_loss(_offset_only, unsup_imgs)
                    if torch.isfinite(cons_loss):
                        loss_terms.append(cons_weight * cons_loss)
                        cons_loss_hist.append(cons_loss.detach().cpu().item())
                    else:
                        # One-shot dump: capture the offending inputs + model output stats
                        # so the NaN source can be inspected offline.
                        if not cons_nan_dumped:
                            cons_nan_dumped = True
                            with torch.no_grad():
                                raw_out = _offset_only(unsup_imgs)
                            stats = (
                                f"images: shape={tuple(unsup_imgs.shape)} "
                                f"min={unsup_imgs.min().item():.4g} max={unsup_imgs.max().item():.4g} "
                                f"finite={torch.isfinite(unsup_imgs).all().item()}; "
                                f"offsets: min={raw_out.min().item():.4g} max={raw_out.max().item():.4g} "
                                f"abs.mean={raw_out.abs().mean().item():.4g} "
                                f"finite={torch.isfinite(raw_out).all().item()}; "
                                f"cons_loss={cons_loss.item()}"
                            )
                            try:
                                import os
                                dump_dir = getattr(args, 'output_path', None)
                                if dump_dir is not None:
                                    dump_path = os.fspath(dump_dir) + "/consistency_nan_dump.pt"
                                    torch.save({
                                        'images': unsup_imgs.detach().cpu(),
                                        'offsets': raw_out.detach().cpu(),
                                        'cons_loss': cons_loss.detach().cpu(),
                                    }, dump_path)
                                    stats += f"  saved to {dump_path}"
                            except Exception as _e:
                                stats += f"  (dump failed: {_e})"
                            warnings.warn(f"consistency_loss is non-finite; dropping this batch's consistency term. {stats}")
                        else:
                            warnings.warn(f"consistency_loss non-finite ({cons_loss.item()}); dropping (suppressed further detail)")

                if not loss_terms:
                    continue
                loss = sum(loss_terms)
            else:
                output = train_model(image_batch)
                loss = train_loss_fn(output, labels.clone()).mean()

        if skip_bad_batches and not torch.isfinite(loss):
            warnings.warn(f"Skipping batch: non-finite loss ({loss.item()})")
            train_optimizer.zero_grad(set_to_none=True)
            # No scaler.update() here: we never called scale()/backward()/step(),
            # so no inf check was recorded. Calling update() would assert.
            skipped_loss += 1
            continue

        train_optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(train_optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)
            if skip_bad_batches and not torch.isfinite(total_norm):
                warnings.warn(f"Skipping batch: non-finite gradient norm ({total_norm.item()})")
                # scaler.step is internally a no-op when infs are present;
                # scaler.update will then reduce the loss scale for next iter.
                scaler.step(train_optimizer)
                scaler.update()
                train_optimizer.zero_grad(set_to_none=True)
                skipped_grad += 1
                continue
            scaler.step(train_optimizer)
            scaler.update()
        else:
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)
            if skip_bad_batches and not torch.isfinite(total_norm):
                warnings.warn(f"Skipping batch: non-finite gradient norm ({total_norm.item()})")
                train_optimizer.zero_grad(set_to_none=True)
                skipped_grad += 1
                continue
            train_optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    if skipped_loss or skipped_grad:
        print(f"[skip_bad_batches] skipped {skipped_loss} batch(es) for non-finite loss, "
              f"{skipped_grad} for non-finite gradient norm")

    mean_loss = float(np.mean(train_loss)) if train_loss else float('nan')
    extra = {
        'mean_sup_loss': float(np.mean(sup_loss_hist)) if sup_loss_hist else float('nan'),
        'mean_cons_loss': float(np.mean(cons_loss_hist)) if cons_loss_hist else float('nan'),
        'n_sup_batches': len(sup_loss_hist),
        'n_cons_batches': len(cons_loss_hist),
    }
    return mean_loss, end - start, extra
    

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

    test_model.eval()
    test_loss = []

    current_f1_list = []
    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels = labels_batch.to(test_device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                output = test_model(image_batch)
                loss = test_loss_fn(output, labels.clone()).mean()
            test_loss.append(loss.detach().cpu().numpy())

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

    f1_array = np.array(current_f1_list)  # either N,2 or N,

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    #  mean_f1 = current_f1_list

    if mean_f1 > best_f1 or save_bool:
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



