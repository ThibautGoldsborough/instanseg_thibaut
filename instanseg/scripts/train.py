import os
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
parser.add_argument('-source', '--source_dataset', default="all", type=str, help = "Which datasets to use for training. Input is 'all' or a list of datasets (e.g. [TNBC_2018,LyNSeC,IHC_TMA,CoNSeP])")
parser.add_argument("-m_f", "--model_folder", type=str, default=None, help = "Name of the model to resume training. This must be a folder inside model_path")
parser.add_argument("-m_p", "--model_path", type=str, default=r"../models", help = "Path to the folder containing the models")
parser.add_argument("-o_p", "--output_path", type=str, default=r"../models", help = "Path to the folder where the results will be saved")
parser.add_argument("-e_s", "--experiment_str", type=str, default="my_first_instanseg", help = "String to identify the experiment")
parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('-num_workers', '--num_workers', default=3, type=int, help = "Number of CPU cores to use for data loading")
parser.add_argument('-ci', '--channel_invariant', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to add a channel invariant model to the pipeline")
parser.add_argument('-target', '--target_segmentation', default="N",type=str, help = " Cells or nuclei or both? Accepts: C,N, NC")  
parser.add_argument('-pixel_size', '--requested_pixel_size', default=None, type=float, help = "Requested pixel size to rescale the input images")

#advanced usage
parser.add_argument("-bs", "--batch_size", type=int, default=3)
parser.add_argument("-e", "--num_epochs", type=int, default=500)
parser.add_argument('-len_epoch', '--length_of_epoch', default=1000, type=int, help = "Number of training samples per epoch")
parser.add_argument('-len_eval', '--length_of_eval', default=200, type=int, help = "Number of validation samples per epoch")
parser.add_argument("-lr", "--lr", type=float, default=0.001, help = "Learning rate")
parser.add_argument("-optim", "--optimizer", type=str, default="adam", help = "Optimizer to use, adam, sgd or adamw")
parser.add_argument("-m", "--model_str", type=str, default="InstanSeg_UNet", help = "Model backbone to use")
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
parser.add_argument("-tf", "--transform_intensity", type=float, default=0.5, help = "Intensity transformation factor")
parser.add_argument("-dim_in", "--dim_in", type=int, default=3,help="Number of channels that the (backbone) model expects. This is also the number of channels a channel invariant model would output.")
parser.add_argument("-dummy", "--dummy", default=False, type=lambda x: (str(x).lower() == 'true'),help="Use the training set as a validation set, this will trigger a warning message. use only for debugging")
parser.add_argument('-bg_weight', '--bg_weight', default=None, type= float, help = "Weight to assign to the background class in the loss function")
parser.add_argument('-instance_loss_fn', '--instance_loss_fn', default="lovasz_hinge", type=str, help = "Loss function to use for instance segmentation: lovasz_hinge or dice_loss are supported. lovasz_hinge is a lot slower to start converging")
parser.add_argument('-seed_loss_fn', '--seed_loss_fn', default="l1_distance", type=str, help = "Loss function to use for seed selection, only ce and l1_distance are supported. ce is much faster, but l1_distance is usually more accurate")
parser.add_argument('-mask_loss_fn', '--mask_loss_fn', default=None, type=str, help = "Loss function to use for the mask channel when dim_seeds=2: ce or dice are supported. If set, dim_seeds is forced to 2.")
parser.add_argument('-anneal', '--cosineannealing', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to use cosine annealing for the learning rate")
parser.add_argument('-o_h', '--optimize_hyperparameters', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to optimize hyperparameters every 10 epochs")
parser.add_argument('-hotstart', '--hotstart_training', default=10, type=int, help = "Number of epochs to train the model with ce before starting the main training loop (default=10)")
parser.add_argument('-window', '--window_size', default=128, type=int, help = "Size of the window containing each instance")
parser.add_argument('-multihead', '--multihead', default= False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to branch the decoder into multiple heads.")
parser.add_argument('-dim_coords', '--dim_coords', default=2, type=int, help = "Dimensionality of the coordinate system. Little support for anything but 2")
parser.add_argument('-dim_seeds', '--dim_seeds', default=1, type=int, help = "Number of seed maps to produce. Little support for anything but 1")
parser.add_argument('-norm', '--norm', default="BATCH", type=str, help = "Norm layer to use: None, INSTANCE, INSTANCE_INVARIANT, BATCH")
parser.add_argument('-mlp_w', '--mlp_width', default=5, type=int, help = "Width of the MLP hidden dim")
parser.add_argument('-augmentation_type', '--augmentation_type', default="minimal", type=str, help = "'minimal' or 'heavy' or 'brightfield_only'")
parser.add_argument('-show_augmentations', '--show_augmentations', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Save 10 example augmented image+label pairs to the output folder")
parser.add_argument('-adaptor_net', '--adaptor_net_str', default="1", type=str, help = "Adaptor net to use")
parser.add_argument('-freeze', '--freeze_main_model', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to freeze the main model")
parser.add_argument('-f_e', '--feature_engineering', default="0", type=str, help = "Feature engineering function to use")
parser.add_argument("-f","--f", default = None, type = str, help = "ignore, this is for jypyter notebook compatibility")
parser.add_argument('-rng_seed', '--rng_seed', default=None, type=int, help = "Optional seed for the random number generator")
parser.add_argument('-use_deterministic', '--use_deterministic', default=False, type=lambda x: (str(x).lower() == 'true'), help = "Whether to use deterministic algorithms (default=False)")
parser.add_argument('-tile', '--tile_size', default=256, type=int, help = "Tile sizes for the input images")
parser.add_argument('-modality', '--modality_filter', default=None, type=str, help = "Filter datasets by image modality (e.g. 'Brightfield', 'Fluorescence'). Default None uses all modalities.")
parser.add_argument('-sampling_mode', '--sampling_mode', default=None, type=str, help="Run embedding clustering instead of training. Options: leiden_dino, leiden_sam")
parser.add_argument('-lora_rank', '--lora_rank', default=0, type=int, help="LoRA rank for SAM/DINO backbone. 0 = disabled, 16 is a good default.")


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


def main(model, loss_fn, train_loader, test_loader, num_epochs=1000, epoch_name='output_epoch',
         prior_train_losses=None, prior_test_losses=None, prior_f1_list=None, prior_f1_list_cells=None, hotstart_epoch=None):
    from instanseg.utils.AI_utils import optimize_hyperparameters, train_epoch, test_epoch
    global best_f1_score, device, method, iou_threshold, args, optimizer, scheduler

    train_losses = list(prior_train_losses) if prior_train_losses else []
    test_losses = list(prior_test_losses) if prior_test_losses else []

    best_f1_score = -1
    f1_list = list(prior_f1_list) if prior_f1_list else []
    f1_list_cells = list(prior_f1_list_cells) if prior_f1_list_cells else []

    for epoch in range(num_epochs):

        print("Epoch:", epoch)

        train_loss, train_time = train_epoch(model, device, train_loader, loss_fn, optimizer, args = args)

        if epoch <= 5 and not args.model_folder:  # Training is just starting AND we are not loading a model
            save_epoch_outputs = True
        else:
            save_epoch_outputs = False

        test_loss, f1_score, test_time = test_epoch(model, device, test_loader, loss_fn, debug=False,
                                                    best_f1=best_f1_score,
                                                    save_bool=save_epoch_outputs,
                                                    args = args,
                                                    postprocessing_fn = method.postprocessing,
                                                    method = method,
                                                    iou_threshold = iou_threshold,
                                                    save_str=str(
                                                        args.output_path / str(
                                                            f"epoch_outputs/{epoch_name}_" + str(epoch))))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 ==0 and args.optimize_hyperparameters:
            best_params = optimize_hyperparameters(model,postprocessing_fn = method.postprocessing,data_loader= test_loader,verbose = not args.on_cluster, show_progressbar = not args.on_cluster)
            method.update_hyperparameters(best_params)


        dict_to_print = {"train_loss": train_loss, "test_loss": test_loss, "training_time": int(train_time),
                         "testing_time": int(test_time)}

        if args.cells_and_nuclei:
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

                
        if f1_score > best_f1_score or save_epoch_outputs:
            best_f1_score = np.maximum(f1_score, best_f1_score)

            print("Saving model, best f1_score:", best_f1_score)

            torch.save({
                'f1_score': float(best_f1_score),
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.output_path / "model_weights.pth") 


        # this is where the loss gets printed
        print(", ".join(f"{k}: {v:.5g}" for k, v in dict_to_print.items()))

        save_training_plot(train_losses, test_losses, f1_list, f1_list_cells,
                           args.output_path, args.cells_and_nuclei, hotstart_epoch=hotstart_epoch)

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
    from instanseg.utils.data_loader import _read_images_from_pth, get_loaders

    args.data_path = Path(args.data_path)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    args.output_path = Path(args.output_path) / args.experiment_str
    print("Saving results to {}".format(os.path.abspath(args.output_path)))
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    os.environ["INSTANSEG_DATASET_PATH"] = os.environ.get("INSTANSEG_DATASET_PATH", str(args.data_path))
    os.environ["INSTANSEG_OUTPUT_PATH"] = os.environ.get("INSTANSEG_OUTPUT_PATH", str(args.output_path))

    # Seed as many rngs as we can
    if args.rng_seed:
        print(f'Setting RNG seed to {args.rng_seed}')
        torch.manual_seed(args.rng_seed)
        np.random.seed(args.rng_seed)
        import random
        random.seed(args.rng_seed)
    else:
        print('RNG seed not set')

    if args.use_deterministic:
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
        
    device = _choose_device(args.device)

    if args.loss_function == "instanseg_loss":
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
                        cells_and_nuclei=args.cells_and_nuclei,
                        window_size = args.window_size,
                        dim_coords= args.dim_coords,
                        dim_seeds = args.dim_seeds,
                        feature_engineering_function=args.feature_engineering,
                        bg_weight = args.bg_weight,
                        mask_loss_fn = args.mask_loss_fn)

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

    model = build_model_from_dict(args_dict, random_seed=args.rng_seed)


    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, torch.randn(1,3,256,256))
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

    if args.loss_function in ["instanseg_loss"]:
        from instanseg.utils.loss.instanseg_loss import has_pixel_classifier_model

        if not has_pixel_classifier_model(model):
            model = method.initialize_pixel_classifier(model, MLP_width = args.mlp_width)

    if args.model_folder:
        if args.model_folder == "None":
            args.model_folder = ""

        model, model_dict = load_model_weights(model, path=args.model_path, folder=args.model_folder, device=device, dict = args_dict)
        

        if not args.channel_invariant:
            optimizer = get_optimizer(model.parameters(),args)
            optimizer.load_state_dict(model_dict['optimizer_state_dict'])

        print("Resuming training from epoch", model_dict['epoch'])

    else:
        # Enable LoRA before creating optimizer (when no hotstart to handle it)
        if args.lora_rank > 0 and args.hotstart_training == 0 and hasattr(model, 'enable_lora'):
            model.freeze_backbone()
            model.enable_lora(rank=args.lora_rank)
            model.unfreeze_backbone()
            optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
        else:
            optimizer = get_optimizer(model.parameters(), args)

    if args.channel_invariant:
        from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
        if not has_AdaptorNet(model):
            model = AdaptorNetWrapper(model,adaptor_net_str=args.adaptor_net_str, norm = args.norm)

        if args.freeze_main_model == True:
            params = model.model.AdaptorNet.parameters()
        else:
            params = model.parameters()

        optimizer = get_optimizer(params, args)

    if args.cosineannealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    
    elif args.model_str.lower() == "cellposesam":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_schedule(epoch, warmup_epochs=10, max_epochs=args.num_epochs):
            if epoch < warmup_epochs:
                # Linear warmup from 0 to max_lr
                return (epoch + 1) / warmup_epochs
            elif epoch < max_epochs - 150:
                # Max LR after warmup
                return 1.0
            elif epoch < max_epochs - 50:
                # First decay stage (reduce by factor of 10)
                return 0.1
            else:
                # Second decay stage (reduce by another factor of 10)
                return 0.01

        # Scheduler with the custom learning rate function
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch))
    else:
        scheduler = None


    if "[" in args.source_dataset:
 
        args.source_dataset = [i.lower() for i in args.source_dataset.replace("[","").replace("]","").replace("'","").split(",")]
        print(type(args.source_dataset), args.source_dataset)
    else:
        args.source_dataset = args.source_dataset.lower()


    train_images, train_labels, train_meta, val_images, val_labels, val_meta = _read_images_from_pth(data_path = args.data_path, 
                                                                                                     dataset = args.dataset,
                                                                                                       data_slice = args.data_slice, 
                                                                                                       dummy = args.dummy, 
                                                                                                       args = args, 
                                                                                                       sets= ["Train","Validation"], 
                                                                                                       complete_dataset=segmentation_dataset)

    train_loader, test_loader = get_loaders(train_images, train_labels, val_images, val_labels, train_meta, val_meta, args)

    if args.show_augmentations:
        from instanseg.utils.visualization import show_images
        aug_dir = args.output_path / "augmentation_examples"
        os.makedirs(aug_dir, exist_ok=True)
        train_dataset = train_loader.dataset
        for i in range(min(10, len(train_dataset))):
            raw_img = train_dataset.X[i]
            raw_lbl = train_dataset.Y[i]
            aug_img, aug_lbl = train_dataset[i][:2]
            show_images(raw_img, raw_lbl, aug_img, aug_lbl,
                        titles=["Raw image", "Raw label", "Aug image", "Aug label"],
                        labels=[1, 3], n_cols=4,
                        save_str=str(aug_dir / f"sample_{i:02d}"))
        print(f"Saved {min(10, len(train_dataset))} augmentation examples to {aug_dir}")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    print(f"CUDA devices visible: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

    if args.sampling_mode is not None:
        from instanseg.utils.sampling import run_sampling
        run_sampling(args, train_loader, train_meta, device)
        # Rebuild loaders with Leiden-weighted sampling and continue to training
        args.weight = True
        train_loader, test_loader = get_loaders(train_images, train_labels, val_images, val_labels, train_meta, val_meta, args)

    if args.save:
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        if not os.path.exists(args.output_path / "epoch_outputs"):
            os.mkdir(args.output_path / "epoch_outputs")
        else:
            import glob
            for f in glob.glob(str(args.output_path / "epoch_outputs" / "*.png")):
                os.remove(f)


    pd.DataFrame.from_dict(args_dict, orient='index').to_csv(args.output_path / "experiment_log.csv",
                                                            header=False)
    
    iou_threshold = np.linspace(0.5, 1.0, 10)

    if args.hotstart_training > 0:
        hot_epochs = args.hotstart_training
        hotstart_lr = 1e-3
        mask_str = f", mask_loss=ce" if args.mask_loss_fn is not None else ""
        print(f"Hotstart for {hot_epochs} epochs with seed_loss=l1_distance, instance_loss=dice_loss{mask_str}, lr={hotstart_lr}")

        method.update_seed_loss("l1_distance")
        method.update_instance_loss("dice_loss")
        if args.mask_loss_fn is not None:
            method.update_mask_loss("ce")

        # Freeze pretrained backbone weights during hotstart
        _backbone_frozen = hasattr(model, 'freeze_backbone')
        if _backbone_frozen:
            print("Freezing backbone weights for hotstart")
            model.freeze_backbone()
            optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args, lr=hotstart_lr)
        else:
            optimizer = get_optimizer(model.parameters(), args, lr=hotstart_lr)

        model, train_losses, test_losses, f1_list, f1_list_cells = main(model, loss_fn, train_loader, test_loader, num_epochs=hot_epochs, epoch_name='hotstart_epoch')

        # Unfreeze backbone weights after hotstart
        if _backbone_frozen:
            if args.lora_rank > 0 and hasattr(model, 'enable_lora'):
                print(f"Enabling LoRA (rank={args.lora_rank}) for main training")
                model.enable_lora(rank=args.lora_rank)
                model.unfreeze_backbone()
            else:
                print("Unfreezing backbone weights for main training")
                model.unfreeze_backbone()

        optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)

        mask_str = f", mask_loss={args.mask_loss_fn}" if args.mask_loss_fn is not None else ""
        print(f"Starting main training with seed_loss={args.seed_loss_fn}, instance_loss={args.instance_loss_fn}{mask_str}, lr={args.lr}")
        method.update_seed_loss(args.seed_loss_fn)
        method.update_instance_loss(args.instance_loss_fn)
        if args.mask_loss_fn is not None:
            method.update_mask_loss(args.mask_loss_fn)

    if args.hotstart_training > 0:
        model, train_losses, test_losses, f1_list, f1_list_cells = main(model, loss_fn, train_loader, test_loader, num_epochs=num_epochs,
            prior_train_losses=train_losses, prior_test_losses=test_losses,
            prior_f1_list=f1_list, prior_f1_list_cells=f1_list_cells,
            hotstart_epoch=args.hotstart_training)
    else:
        model, train_losses, test_losses, f1_list, f1_list_cells = main(model, loss_fn, train_loader, test_loader, num_epochs=num_epochs)

    from instanseg.utils.model_loader import load_model
    model, model_dict = load_model(folder="", path=args.output_path) #Load model from checkpoint
    model.eval()
    model.to(device)

    df = pd.DataFrame({"train_loss": train_losses, "test_loss": test_losses, "f1_score": f1_list})
    df.to_csv(args.output_path / "experiment_metrics.csv", index=False, header=True)


    fig = plot_average(train_losses, test_losses, window_size=len(train_losses) // 10 + 1)
    plt.savefig(args.output_path / "loss.png")
    plt.close()

    if args.cells_and_nuclei:
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