
import numpy as np
import warnings

def _keep_images(item, args):

    #args.source_dataset = str(args.source_dataset).lower().replace("[","").replace("]","").replace("'","").split(",")

    if args.source_dataset != ["all"] and item[
        'parent_dataset'].lower() not in args.source_dataset:  # remove items that are not of the desired dataset
        return False
    elif 'duplicate' in item.keys() and item['duplicate']:  # remove items that are duplicates
        return False
    elif args.target_segmentation == "N" and "nucleus_masks" not in item.keys():
        return False
    elif args.target_segmentation == "C" and "cell_masks" not in item.keys():
        return False
    else:
        return True
    
 
def _format_labels(item,target_segmentation):
            
    if "cell_masks" in item.keys():
        item["cell_masks"] = get_image(item["cell_masks"])
    
    if "nucleus_masks" in item.keys():
        item["nucleus_masks"] = get_image(item["nucleus_masks"])
 
    elif "masks" in item.keys():
        item["nucleus_masks"] = get_image(item["masks"])
    
    if target_segmentation == "N":
        if "nucleus_masks" not in item.keys():
            c,h,w = item['image'].shape
            labels = np.zeros((h,w)) -1
        else:
            labels = item["nucleus_masks"]
    elif target_segmentation == "C":
        if "cell_masks" not in item.keys():
            c,h,w = item['image'].shape
            labels = np.zeros((h,w)) -1
        else:
            labels = item["cell_masks"]
    elif "N" in target_segmentation and "C" in target_segmentation:
            if "nucleus_masks" in item.keys() and "cell_masks" in item.keys():
                labels = np.stack((item["nucleus_masks"], item["cell_masks"]))
            elif "nucleus_masks" in item.keys() and "cell_masks" not in item.keys():
                labels = item['nucleus_masks']
                labels = np.stack((labels, np.zeros_like(labels).astype(np.int32) - 1))
            elif "nucleus_masks" not in item.keys() and "cell_masks" in item.keys():
                labels = item['cell_masks']
                labels = np.stack((np.zeros_like(labels).astype(np.int32) - 1, labels))
            else:
                raise NotImplementedError("No labels found")
    else:
        raise NotImplementedError("Target segmentation not recognized", target_segmentation)


    return labels
 


def export_dataset_dict_as_folder(dataset,destination = "benchmarking_data"):
    from collections import defaultdict
    from tqdm import tqdm
    import pandas as pd
    import torch
    import tifffile
    import os
    from instanseg.utils.utils import _move_channel_axis
    from pathlib import Path

    expanded_datasets = Path("../datasets/") / destination


    dataset_dfs = defaultdict(lambda: defaultdict(lambda: list()))

    for fold_name, fold in dataset.items():
        for i, data_info in tqdm(enumerate(fold), ncols=100, total=len(fold)):
            img = data_info['image']
            dataset_name = data_info['parent_dataset']
            if dataset_name == "Aleynik" or dataset_name == "TissueNet":
                continue
            if "nucleus_masks" not in data_info.keys():
                mask = data_info['masks']
            else:
                mask = data_info['nucleus_masks']
            storage_folder = os.path.join(expanded_datasets, dataset_name, fold_name)
            if not os.path.exists(storage_folder):
                os.makedirs(storage_folder)
            tifffile.imwrite(os.path.join(storage_folder, f"img{i}.tiff"), _move_channel_axis(img))
            tifffile.imwrite(os.path.join(storage_folder, f"img{i}_mask.tiff"), mask)
            for key in data_info.keys():
                if key not in ['masks','image', 'nucleus_masks', 'cell_masks']:
                    dataset_dfs[dataset_name][key].append(data_info[key])
            dataset_dfs[dataset_name]["ID"].append(i)


    for df_name, df in dataset_dfs.items():
        print(df_name)
        pd.DataFrame(df).to_csv(os.path.join(expanded_datasets, df_name, f"{df_name}_metadata.csv"))


def get_image(img_object):
    import tifffile
    import os
    from pathlib import Path

    if type(img_object) == str:

        data_path = os.environ["INSTANSEG_DATASET_PATH"]

        img_path = Path(os.path.join(data_path,img_object))

        if Path(img_path).exists():
            img = tifffile.imread(img_path)
            return img
        else:
           
            if Path(str(Path(img_path).parents[1]) + ".zip").exists():
                import shutil
                import os
                print("Inflating zip file")

                print((str(Path(img_path).parents[1]) + ".zip"))

                shutil.unpack_archive(str(Path(img_path).parents[1]) + ".zip", Path(img_path).parents[2])
            
            #breakpoint()
            img = tifffile.imread(img_path)
            return img
    else:
        return img_object



def _read_images_from_path(data_path= "../datasets", 
                          dataset = "segmentation", 
                          data_slice = None, 
                          dummy = False, 
                          args = None, 
                          sets = ["Train","Validation"], 
                          ):
    
    from pathlib import Path
    import os
    import skimage.io

    datasets_available = sorted(os.listdir(data_path))
    print("Datasets available ", datasets_available)

    source_dataset = args.source_dataset

    assert len(sets) !=2, "Only one set can be loaded at a time"
    data_dicts = {}

    source_dataset = str(args.source_dataset).lower().replace("[","").replace("]","").replace("'","").split(",")


    for folder in datasets_available:
        if folder.lower() in source_dataset:
            dataset_path = Path(data_path) / folder
            for _set in sets:

                if _set not in data_dicts.keys():
                    data_dicts[_set] = [[],[],[]]

                for image_str in sorted(os.listdir(dataset_path / f"{_set}")):
                    if "mask" in image_str:
                        continue
                    image = skimage.io.imread(dataset_path / f"{_set}/{image_str}")
                    

                    mask_str = image_str.replace(".tiff","_mask.tiff")

                    mask = skimage.io.imread(dataset_path / f"{_set}/{mask_str}")


                    meta = {"parent_dataset": folder, "modality": "Brightfield", "pixel_size": None, "name": image_str}

                    data_dicts[_set][0].append(image)
                    data_dicts[_set][1].append(mask)
                    data_dicts[_set][2].append(meta)
                  #  breakpoint()


    return_list = []
    for _set in sets:
        return_list.extend(data_dicts[_set])

        assert len(data_dicts[_set][0]) > 0, "No images in the dataset meet the requirements. (Hint: Check that the source argument is correct)"
    
    return return_list
   # breakpoint()


def _read_images_from_pth(data_path= "../datasets", dataset = "segmentation", data_slice = None, dummy = False, args = None, sets = ["Train","Validation"], complete_dataset = None):
    from pathlib import Path
    import torch
    import os 

    if complete_dataset is None:
        if not os.environ.get("INSTANSEG_DATASET_PATH"):
            os.environ["INSTANSEG_DATASET_PATH"] = Path(os.path.join(os.path.dirname(__file__),data_path))
        data_path = os.environ["INSTANSEG_DATASET_PATH"]
        if ".pth" in dataset:
            path_of_pth = os.path.join(data_path,dataset)
        else:
            path_of_pth = os.path.join(data_path,str(dataset + "_dataset.pth"))

        print("Loading dataset from ", os.path.abspath(path_of_pth))

        try:
            complete_dataset = torch.load(path_of_pth,weights_only = False)
        except:
            complete_dataset = torch.load(path_of_pth)
    

    data_dicts = {}

    for _set in sets:
        print("Datasets available in ", _set)
        unique_values, counts = np.unique([item['parent_dataset'] for item in complete_dataset[_set]], return_counts=True)
        print(set(zip(unique_values, counts)))

        data_dicts[_set] = []
        images_local = [get_image(item['image']) for item in complete_dataset[_set] if _keep_images(item, args)][:data_slice]
 
        labels_local = [_format_labels(item,target_segmentation = args.target_segmentation) for item in complete_dataset[_set] if _keep_images(item, args)][:data_slice]
        metadata = [{k: v for k, v in item.items() if k not in ('image', 'cell_masks','nucleus_masks', 'class_masks')} for item in complete_dataset[_set] if _keep_images(item, args)][:data_slice]

        data_dicts[_set].extend([images_local,labels_local,metadata])

        print("After filtering using:")
        unique_values, counts = np.unique([item['parent_dataset'] for item in data_dicts[_set][2]], return_counts=True)
        print(set(zip(unique_values, counts)))

    if dummy:
        warnings.warn("Using same train and validation sets !")
        data_dicts["Validation"] = data_dicts["Train"]

    return_list = []
    for _set in sets:
        return_list.extend(data_dicts[_set])

        assert len(data_dicts[_set][0]) > 0, "No images in the dataset meet the requirements. (Hint: Check that the source argument is correct)"

    return return_list


def get_loaders(train_images_local, train_labels_local, val_images_local, val_labels_local, train_meta, val_meta, args):
    from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
    from instanseg.utils.augmentation_config import get_augmentation_dict
    from instanseg.utils.AI_utils import Segmentation_Dataset, collate_fn
    from torch.utils.data import DataLoader
    from instanseg.utils.utils import count_instances


    if args.rng_seed is not None:
        import torch
        torch.manual_seed(args.rng_seed)

    augmentation_dict = get_augmentation_dict(args.dim_in, 
                                              nuclei_channel=None, 
                                              amount=args.transform_intensity,
                                              pixel_size=args.requested_pixel_size,
                                              mean_diameter=args.mean_object_diameter, 
                                              augmentation_type=args.augmentation_type)

    train_data = Segmentation_Dataset(train_images_local, 
                                      train_labels_local, 
                                      metadata=train_meta,
                                      size=(args.tile_size, args.tile_size), 
                                      augmentation_dict=augmentation_dict['train'],
                                      debug=False,
                                      dim_in=args.dim_in,
                                      cells_and_nuclei=args.cells_and_nuclei,
                                      random_seed=args.rng_seed,
                                      target_segmentation=args.target_segmentation, 
                                      channel_invariant = args.channel_invariant)

    test_data = Segmentation_Dataset(val_images_local, val_labels_local, 
                                     size=(args.tile_size, args.tile_size), 
                                     metadata=val_meta,
                                     dim_in=args.dim_in,
                                     augmentation_dict=augmentation_dict['test'],
                                     random_seed = args.rng_seed,
                                     cells_and_nuclei=args.cells_and_nuclei,
                                     target_segmentation=args.target_segmentation,
                                     channel_invariant = args.channel_invariant)

    test_sampler = RandomSampler(test_data,num_samples=int(
                args.length_of_epoch * 0.2))


    if not args.weight:
        if args.length_of_epoch is not None:
            train_sampler = RandomSampler(train_data, num_samples=args.length_of_epoch)
            test_sampler = RandomSampler(test_data, num_samples=int(
                args.length_of_epoch * 0.2))  # This is relates to the standard 80/20 split
        else:
            train_sampler = RandomSampler(train_data)

    else:

        datasets_train = [meta["parent_dataset"] for meta in train_meta]
        datasets,counts = np.unique(datasets_train,return_counts=True)
        dict_datasets = dict(zip(datasets,counts / sum(counts)))
        freq = [ 1/ dict_datasets[dataset] for dataset in datasets_train]
        rel_freq = (freq / sum(freq))
        if args.length_of_epoch is not None:
            train_sampler = WeightedRandomSampler(rel_freq, args.length_of_epoch)
        else:
            train_sampler = WeightedRandomSampler(rel_freq, len(freq))

        print(dict_datasets)

        datasets_val = [meta["parent_dataset"] for meta in val_meta]
        datasets,counts = np.unique(datasets_val,return_counts=True)
        dict_datasets = dict(zip(datasets,counts / sum(counts)))
        freq = [ 1/ dict_datasets[dataset] for dataset in datasets_val]
        rel_freq = (freq / sum(freq))

        if args.length_of_epoch is not None:
            test_sampler = WeightedRandomSampler(rel_freq, int(args.length_of_epoch * 0.2))
        else:
            test_sampler = WeightedRandomSampler(rel_freq, len(freq))


    train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=train_sampler, persistent_workers=True)
    test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers,
                             sampler=test_sampler, persistent_workers=True)

    return train_loader, test_loader
