import numpy as np


def build_monai_model(model_str: str, build_model_dictionary: dict):


    if model_str == "AttentionUNet":
        from monai.networks.nets import AttentionUnet

        model = AttentionUnet(spatial_dims=2, in_channels=int(build_model_dictionary["dim_in"]),
                              out_channels=build_model_dictionary["dim_out"], \
                              dropout=build_model_dictionary["dropprob"], channels=build_model_dictionary["layers"], \
                              strides=tuple([2 for _ in build_model_dictionary["layers"][:-1]])
                              )
    elif model_str == "FlexibleUNet":
        from monai.networks.nets import FlexibleUNet
        model = FlexibleUNet(in_channels=build_model_dictionary["dim_in"],
                             out_channels=build_model_dictionary["dim_out"], dropout=build_model_dictionary["dropprob"],
                             backbone="efficientnet-b0")
        

    elif model_str == "BasicUNetPlusPlus":
        from monai.networks.nets import BasicUNetPlusPlus
        model = BasicUNetPlusPlus(spatial_dims=2, in_channels=build_model_dictionary["dim_in"],
                                  out_channels=build_model_dictionary["dim_out"],
                                  dropout=build_model_dictionary["dropprob"])

        class ModelWrapper(BasicUNetPlusPlus):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, inputs):
                output = super().forward(inputs)
                modified_output = output[0]  # Modify the output here as needed
                return modified_output

        model = ModelWrapper(spatial_dims=2, in_channels=build_model_dictionary["dim_in"],
                             out_channels=build_model_dictionary["dim_out"], dropout=build_model_dictionary["dropprob"])

    elif model_str == "UNETR":
        from monai.networks.nets import UNETR
        model = UNETR(in_channels=build_model_dictionary["dim_in"], out_channels=build_model_dictionary["dim_out"],
                      img_size=256, feature_size=32, norm_name='batch', spatial_dims=2)


    else:
        raise NotImplementedError("Model not implemented: " + model_str)

    return model


def read_model_args_from_csv(path=r"../results/", folder=""):
    import pandas as pd
    from pathlib import Path
    model_path = Path(path) / folder
    df = pd.read_csv(model_path / "experiment_log.csv", header=None)
    build_model_dictionary = dict(zip(list(df[0]), list(df[1])))

    if "model_shape" in build_model_dictionary.keys():
        build_model_dictionary["model_shape"] = eval(build_model_dictionary["model_shape"])
    for key in ["dim_in", "n_sigma", "dim_out", "dim_coords"]:
        build_model_dictionary[key] = eval(str(build_model_dictionary[key])) if str(
            build_model_dictionary[key]) != "nan" else None

    if "dropprob" in build_model_dictionary.keys():
        build_model_dictionary["dropprob"] = float(build_model_dictionary["dropprob"])
    if "layers" in build_model_dictionary.keys():
        build_model_dictionary["layers"] = tuple(eval(build_model_dictionary["layers"]))
    if "requested_pixel_size" in build_model_dictionary.keys():
        build_model_dictionary["pixel_size"] = float(build_model_dictionary["requested_pixel_size"])
    if "cells_and_nuclei" in build_model_dictionary.keys():
        build_model_dictionary["cells_and_nuclei"] = bool(eval(build_model_dictionary["cells_and_nuclei"]))
    if "norm" in build_model_dictionary.keys():
        if build_model_dictionary["norm"] == "None" or str(build_model_dictionary["norm"]).lower() == "nan":
            build_model_dictionary["norm"] = None
        else:
            build_model_dictionary["norm"] = str(build_model_dictionary["norm"])
    else:
        print("Norm not specified in model dictionary")
        build_model_dictionary["norm"] = None
    if "feature_engineering" in build_model_dictionary.keys():
        build_model_dictionary["feature_engineering"] = str(build_model_dictionary["feature_engineering"])
    else:
        print("Feature engineering not specified in model dictionary")
        build_model_dictionary["feature_engineering"] = "0"
    if "adaptor_net_str" in build_model_dictionary.keys():
        build_model_dictionary["adaptor_net_str"] = str(build_model_dictionary["adaptor_net_str"])
    if "multihead" in build_model_dictionary.keys():
        build_model_dictionary["multihead"] = bool(eval(build_model_dictionary["multihead"]))
    else:
        build_model_dictionary["multihead"] = False
    if "channel_invariant" in build_model_dictionary.keys():
        build_model_dictionary["channel_invariant"] = bool(eval(build_model_dictionary["channel_invariant"]))
    if "only_positive_labels" in build_model_dictionary.keys():
        build_model_dictionary["only_positive_labels"] = bool(eval(build_model_dictionary["only_positive_labels"]))
    else:
        build_model_dictionary["only_positive_labels"] = True
    if "dim_seeds" in build_model_dictionary.keys():
        build_model_dictionary["dim_seeds"] = int(build_model_dictionary["dim_seeds"])
    else:
        build_model_dictionary["dim_seeds"] = 1
    if "seed_merging" in build_model_dictionary.keys():
        build_model_dictionary["seed_merging"] = bool(eval(str(build_model_dictionary["seed_merging"])))
    else:
        build_model_dictionary["seed_merging"] = False

    return build_model_dictionary


def build_model_from_dict(build_model_dictionary, random_seed = None):

    #set seed 
    if random_seed is not None:
        import torch
        torch.manual_seed(random_seed)

    if build_model_dictionary["dim_in"] == 0 or build_model_dictionary["dim_in"] is None:
        dim_in = 3  # Channel invariance currently outputs a 3 channel image
    else:
        dim_in = build_model_dictionary["dim_in"]

    if "dropprob" not in build_model_dictionary.keys():
        build_model_dictionary["dropprob"] = 0.0

    if build_model_dictionary["model_str"].lower() == "instanseg_unet":
            from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet
            print("Generating InstanSeg_UNet")
            multihead = build_model_dictionary["multihead"]

            if build_model_dictionary["cells_and_nuclei"]:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    from itertools import chain
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
                    out_channels = list(chain(*out_channels))
                
                else:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
            else:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds]]
                else:
                    out_channels = [[build_model_dictionary["dim_coords"]], [build_model_dictionary["n_sigma"]],[n_seeds]]

            model = InstanSeg_UNet(in_channels=dim_in, 
                            layers = np.array(build_model_dictionary["layers"])[::-1],
                            out_channels=out_channels,
                            norm  = build_model_dictionary["norm"], 
                            dropout=build_model_dictionary["dropprob"])


    elif (build_model_dictionary["model_str"].lower() in {"maxvit_pico", "maxvit_tiny", "maxvit_base", "maxvit_large"}
          or build_model_dictionary["model_str"].lower().startswith(("maxvit_", "maxxvit"))):
            from instanseg.utils.models.MaxViT import MaxViT, maxvit_pico, maxvit_tiny, maxvit_base, maxvit_large
            maxvit_name = build_model_dictionary["model_str"].lower()
            print(f"Generating {maxvit_name}")
            multihead = build_model_dictionary["multihead"]
            # AdaLN conditioning: the model predicts a SINGLE segmentation and is
            # told (per-sample, via an embedding) whether it is cell ("C") or
            # nucleus ("N"). So even when cells_and_nuclei is set on the data
            # side, the model output is single-head. This rule lives here so it
            # is identical on build and on reload (the `adaln` key is persisted
            # in the model dict).
            adaln = bool(build_model_dictionary.get("adaln", False))
            n_seeds = build_model_dictionary["dim_seeds"]

            if build_model_dictionary["cells_and_nuclei"] and not adaln:
                if not multihead:
                    from itertools import chain
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"], n_seeds] for i in range(2)]
                    out_channels = list(chain(*out_channels))
                else:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"], n_seeds] for i in range(2)]
            else:
                if not multihead:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"], n_seeds]]
                else:
                    out_channels = [[build_model_dictionary["dim_coords"]], [build_model_dictionary["n_sigma"]], [n_seeds]]

            import torch as _torch  # local alias to avoid leaking into module scope
            _use_fp16 = bool(build_model_dictionary.get("fp16", False))
            common_kwargs = dict(
                in_channels=dim_in,
                out_channels=out_channels,
                norm=build_model_dictionary["norm"],
                dropout=build_model_dictionary["dropprob"],
                attn_dropout=build_model_dictionary["dropprob"],
                drop_path_rate=float(build_model_dictionary.get("drop_path_rate", 0.0)),
                # When the outer training loop uses fp16 autocast (AI_utils
                # wraps forward in `torch.amp.autocast(..., dtype=float16)`
                # when `--fp16` is set), make sure MaxViT's internal amp_dtype
                # matches, so any nested autocast is a no-op instead of a
                # dtype-switching mid-forward mismatch.
                amp_dtype=_torch.float16 if _use_fp16 else _torch.bfloat16,
                compile=bool(build_model_dictionary.get("compile", False)),
                compile_mode=build_model_dictionary.get("compile_mode", "default"),
            )
            if adaln:
                # Cell vs nucleus conditioning. Classes are ("N","C") so channel
                # 0 (nucleus) -> "N" and channel 1 (cell) -> "C", matching the
                # label channel order produced by data_loader._format_labels.
                # Small shared bottleneck (cond_dim, default 8) for stable,
                # low-rank conditioning; overridable via the config key.
                common_kwargs["adaln"] = True
                common_kwargs["adaln_classes"] = ("N", "C")
                common_kwargs["adaln_cond_dim"] = int(build_model_dictionary.get("adaln_cond_dim", 8))
            # Preset names match timm's size names (pico/tiny/base/large). Legacy
            # checkpoints used maxvit/maxvit_tiny (=pico) and maxvit_base
            # (=timm tiny); their experiment_log.csv files were migrated to
            # maxvit_pico / maxvit_tiny, so only the new preset names are handled
            # by the builders. Any other maxvit_*/maxxvit* string is treated as a
            # raw timm backbone name and passed straight to MaxViT (the build path
            # reads embed_dim/depths/stem_width generically from the timm cfg).
            preset_builders = {
                "maxvit_pico": maxvit_pico,
                "maxvit_tiny": maxvit_tiny,
                "maxvit_base": maxvit_base,
                "maxvit_large": maxvit_large,
            }
            if maxvit_name in preset_builders:
                model = preset_builders[maxvit_name](**common_kwargs)
            else:
                # Heavy backbones need gradient checkpointing to fit in memory,
                # mirroring the maxvit_large preset. Allow an explicit override
                # via the experiment config key `gradient_checkpointing`.
                _gc = bool(build_model_dictionary.get(
                    "gradient_checkpointing",
                    any(s in maxvit_name for s in ("base", "large", "xlarge")),
                ))
                print(f"  raw timm backbone, gradient_checkpointing={_gc}")
                model = MaxViT(timm_name=maxvit_name,
                               gradient_checkpointing=_gc, **common_kwargs)


    elif build_model_dictionary["model_str"].lower() == "instanseg_sam":
            from instanseg.utils.models.InstanSeg_SAM import InstanSeg_SAM
            print("Generating InstanSeg_SAM")
            multihead = build_model_dictionary["multihead"]

            if build_model_dictionary["cells_and_nuclei"]:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    from itertools import chain
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
                    out_channels = list(chain(*out_channels))

                else:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
            else:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds]]
                else:
                    out_channels = [[build_model_dictionary["dim_coords"]], [build_model_dictionary["n_sigma"]],[n_seeds]]

            model = InstanSeg_SAM(in_channels=dim_in,
                            layers = np.array(build_model_dictionary["layers"])[::-1],
                            out_channels=out_channels,
                            norm  = build_model_dictionary["norm"],
                            dropout=build_model_dictionary["dropprob"])

    elif build_model_dictionary["model_str"].lower() == "instanseg_dino":
            from instanseg.utils.models.InstanSeg_DINO import InstanSeg_DINO
            print("Generating InstanSeg_DINO")
            multihead = build_model_dictionary["multihead"]

            if build_model_dictionary["cells_and_nuclei"]:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    from itertools import chain
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
                    out_channels = list(chain(*out_channels))

                else:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
            else:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds]]
                else:
                    out_channels = [[build_model_dictionary["dim_coords"]], [build_model_dictionary["n_sigma"]],[n_seeds]]

            model = InstanSeg_DINO(in_channels=dim_in,
                            layers = np.array(build_model_dictionary["layers"])[::-1],
                            out_channels=out_channels,
                            norm  = build_model_dictionary["norm"],
                            dropout=build_model_dictionary["dropprob"])

    elif build_model_dictionary["model_str"].lower() in {"eupe_convnext_tiny", "eupe_convnext_small", "eupe_convnext_base"}:
            from instanseg.utils.models.EUPE import EUPE
            eupe_model = build_model_dictionary["model_str"].lower()
            print(f"Generating EUPE[{eupe_model}]")
            multihead = build_model_dictionary["multihead"]

            if build_model_dictionary["cells_and_nuclei"]:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    from itertools import chain
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
                    out_channels = list(chain(*out_channels))
                else:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds] for i in range(2)]
            else:
                n_seeds = build_model_dictionary["dim_seeds"]
                if not multihead:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],n_seeds]]
                else:
                    out_channels = [[build_model_dictionary["dim_coords"]], [build_model_dictionary["n_sigma"]],[n_seeds]]

            eupe_kwargs = {}
            if "eupe_weights" in build_model_dictionary and str(build_model_dictionary["eupe_weights"]).lower() not in ("nan", "none", ""):
                eupe_kwargs["eupe_weights"] = str(build_model_dictionary["eupe_weights"])
            if "eupe_github_repo" in build_model_dictionary and str(build_model_dictionary["eupe_github_repo"]).lower() not in ("nan", "none", ""):
                eupe_kwargs["eupe_github_repo"] = str(build_model_dictionary["eupe_github_repo"])
            if "eupe_pretrained" in build_model_dictionary and str(build_model_dictionary["eupe_pretrained"]).lower() not in ("nan", ""):
                eupe_kwargs["eupe_pretrained"] = bool(eval(str(build_model_dictionary["eupe_pretrained"]).capitalize()))

            model = EUPE(in_channels=dim_in,
                         out_channels=out_channels,
                         eupe_model=eupe_model,
                         norm=build_model_dictionary["norm"],
                         dropout=build_model_dictionary["dropprob"],
                         **eupe_kwargs)

    elif build_model_dictionary["model_str"].lower() == "cellposesam":
        from instanseg.utils.models.CellposeSam import CellposeSam
        print("Generating CellposeSam")
        model = CellposeSam(nout=build_model_dictionary["dim_out"])
    elif build_model_dictionary["model_str"].lower() == "sam_unet":
        from instanseg.utils.models.CellposeSam import SAM_UNet
        print("Generating SAM_UNet")
        if build_model_dictionary["cells_and_nuclei"]:
            out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],build_model_dictionary["dim_seeds"]] for i in range(2)]
        else:
            out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],build_model_dictionary["dim_seeds"]]]
        model = SAM_UNet(in_channels=dim_in, out_channels=out_channels,
                         layers=np.array(build_model_dictionary["layers"])[::-1],
                         norm=build_model_dictionary["norm"], dropout=build_model_dictionary["dropprob"])
    
    elif build_model_dictionary["model_str"].lower() == "sam":
        from instanseg.utils.models.sam import SAMFeatureExtractor
        print("Generating SAMFeatureExtractor")
        model = SAMFeatureExtractor(nout=build_model_dictionary["dim_out"])
            
    else:
        model = build_monai_model(build_model_dictionary["model_str"], build_model_dictionary)

    return model


def remove_module_prefix_from_dict(dictionary):
    """
    Removes the module prefix from a dictionary of model weights
    :param dictionary: dictionary of model weights
    :return: modified dictionary
    """
    modified_dict = {}
    for key, value in dictionary.items():
        if key.startswith('module.'):
            modified_dict[key[7:]] = value
        else:
            modified_dict[key] = value
    return modified_dict


def has_pixel_classifier_state_dict(state_dict):
    return bool(sum(['pixel_classifier' in key for key in state_dict.keys()]))

def has_adaptor_net_state_dict(state_dict):
    return bool(sum(['AdaptorNet' in key for key in state_dict.keys()]))

def has_pixel_classifier_model(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'pixel_classifier' in module_class or 'ProbabilityNet' in module_class:
                return True
    return False


def duplicate_decoder_heads(model, state_dict):
    """Warm-start the cell branch of an NC model from a single-task (N or C) checkpoint.

    A ``cells_and_nuclei`` (NC) MaxViT has twice as many decoder heads as a
    single-task model: ``heads.0..K-1`` for one task and ``heads.K..2K-1`` for
    the other (see ``model_loader.build_model_from_dict`` / ``_MaxViTDecoder``).
    A checkpoint trained with ``-target N`` only contains ``heads.0..K-1``, so a
    strict load fails with missing keys for the cell heads. This copies each
    existing head ``i`` into the matching missing head ``i + K`` so both task
    branches start from the trained single-task weights.

    Mutates and returns ``state_dict``. Only acts when a decoder's model head
    count is exactly twice the checkpoint's; any other mismatch is left for the
    subsequent strict load to report.

    Parameters
    ----------
    model : torch.nn.Module
        The freshly built (NC) model whose ``state_dict`` defines the target keys.
    state_dict : dict[str, torch.Tensor]
        The checkpoint state dict to augment in place.

    Returns
    -------
    tuple[dict[str, torch.Tensor], bool]
        The (possibly augmented) state dict and whether any head was duplicated.
    """
    import re
    head_re = re.compile(r"^(decoders\.\d+\.heads\.)(\d+)(\..*)$")

    def _heads_by_prefix(keys: "list[str]") -> "dict[str, set[int]]":
        out: dict[str, set[int]] = {}
        for k in keys:
            m = head_re.match(k)
            if m:
                out.setdefault(m.group(1), set()).add(int(m.group(2)))
        return out

    model_keys = set(model.state_dict().keys())
    model_heads = _heads_by_prefix(list(model_keys))
    ckpt_heads = _heads_by_prefix(list(state_dict.keys()))

    duplicated = False
    for prefix, m_idx in model_heads.items():
        c_idx = ckpt_heads.get(prefix, set())
        n_model, n_ckpt = len(m_idx), len(c_idx)
        if n_ckpt == 0 or n_model != 2 * n_ckpt:
            continue  # not the single->double head case; leave it to strict load
        for k in [key for key in state_dict if key.startswith(prefix)]:
            m = head_re.match(k)
            dst_key = f"{prefix}{int(m.group(2)) + n_ckpt}{m.group(3)}"
            if dst_key in model_keys and dst_key not in state_dict:
                state_dict[dst_key] = state_dict[k].clone()
                duplicated = True
    return state_dict, duplicated


def load_model_weights(model, device, folder, path=r"../models/", dict = None):
    import torch
    from pathlib import Path
    model_path = Path(path) / folder
    if torch.cuda.is_available():
        model_dict = torch.load(model_path / "model_weights.pth", weights_only= False)
    else:
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
                print('CUDA not available - attempting to load MPS model')
            else:
                device = 'cpu'
                print('CUDA not available - attempting to load CPU model')
        model_dict = torch.load(model_path / "model_weights.pth", map_location=device)

    model_dict['model_state_dict'] = remove_module_prefix_from_dict(model_dict['model_state_dict'])

    if has_pixel_classifier_state_dict(model_dict['model_state_dict']) and not has_pixel_classifier_model(model):
        from instanseg.utils.loss.instanseg_loss import InstanSeg

        method = InstanSeg(n_sigma=int(dict["n_sigma"]), feature_engineering_function= dict["feature_engineering"],dim_coords = dict["dim_coords"],dim_seeds = dict["dim_seeds"],device =device)
        model = method.initialize_pixel_classifier(model, MLP_width=int(dict["mlp_width"]))
    

    from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
    if has_adaptor_net_state_dict(model_dict['model_state_dict']) and not has_AdaptorNet(model):
        from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
        model = AdaptorNetWrapper(model, norm = dict["norm"],adaptor_net_str = dict["adaptor_net_str"])

    #from instanseg.utils.AI_utils import set_running_stats
    #set_running_stats(model,device = "cuda")

    model_dict['model_state_dict'], _duplicated_heads = duplicate_decoder_heads(
        model, model_dict['model_state_dict'])
    model_dict['duplicated_decoder_heads'] = _duplicated_heads
    if _duplicated_heads:
        print("Warm-starting cell decoder heads by duplicating the single-task "
              "(nuclei) heads from the checkpoint into the NC model.")

    model.load_state_dict(model_dict['model_state_dict'], strict=True)
    model.to(device)

    return model, model_dict

def load_model(folder,path=r"../models/", device='cpu'):
    build_model_dictionary = read_model_args_from_csv(path=path, folder=folder)

    empty_model = build_model_from_dict(build_model_dictionary)

    model, _ = load_model_weights(empty_model, path=path, folder=folder, device=device, dict = build_model_dictionary)

    return model, build_model_dictionary
