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
    skip_bad_batches = getattr(args, 'skip_bad_batches', True)
    skipped_loss = 0
    skipped_grad = 0

    accelerator = getattr(args, "accelerator", None)
    is_main = accelerator.is_main_process if accelerator is not None else True
    device = accelerator.device if accelerator is not None else train_device

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster or not is_main):

        image_batch = image_batch.to(device, non_blocking=True)
        labels = labels_batch.to(device, non_blocking=True)

        output = train_model(image_batch)
        loss = train_loss_fn(output, labels.clone()).mean()

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
            continue

        train_optimizer.zero_grad()
        if accelerator is not None:
            accelerator.backward(loss)
            total_norm = accelerator.clip_grad_norm_(train_model.parameters(), args.clip)
        else:
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        # accelerator.clip_grad_norm_ returns the synced (global) norm, so the
        # skip decision below is already identical on every rank — no extra
        # all-reduce needed here.
        if skip_bad_batches and not torch.isfinite(total_norm):
            warnings.warn(f"Skipping batch: non-finite gradient norm ({total_norm.item()})")
            train_optimizer.zero_grad(set_to_none=True)
            skipped_grad += 1
            continue
        train_optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

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

    current_f1_list = []
    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster or not is_main):
            image_batch = image_batch.to(device, non_blocking=True)
            labels = labels_batch.to(device, non_blocking=True)
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



