from monai.transforms import (
    AsDiscrete,
    # AddChanneld,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    SqueezeDimd,
    Lambda,
    CastToTyped
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import json

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..")
# from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['post_label'] = data[0]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data, lazy=False):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data, lazy=False):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data, lazy=False):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key not in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_


# test_16task_names = ['s0867',
#  's1339',
#  's0070',
#  's0909',
#  's0359',
#  's0153',
#  's1071',
#  's0213',
#  's0708',
#  's0563',
#  's0710',
#  's0910',
#  's0913',
#  's0262',
#  's0876',
#  's0040',
#  's0266',
#  's0491',
#  's1068',
#  's0875']

def get_loader(args):
    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            # Lambda(func=lambda x: x[0] if isinstance(x, list) else x),  # 解包列表
            Lambda(func=lambda x: torch.cat(x, dim=0) if isinstance(x, list) else x),  # 解包列表
            
            CastToTyped(keys=["image"], dtype=torch.float32),
            CastToTyped(keys=["label", "post_label"], dtype=torch.int8),
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []

    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_train.txt'):
            name = line.strip().split()[1].split('.')[0]
            train_img.append(args.data_root_path + line.strip().split()[0])
            train_lbl.append(args.data_root_path + line.strip().split()[1])
            train_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            train_name.append(name)
    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))


    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_val.txt'):

            if '15_3DIRCADb' in line:
                name = line.strip().split()[1][:-7]
                print(name)
            else:
                name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)

    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))


    ## test dict part
    test_img = []
    test_lbl = []
    test_post_lbl = []
    test_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_test.txt'):
            if '15_3DIRCADb' in line:
                name = line.strip().split()[1][:-7]
                print(name)
            else:
                name = line.strip().split()[1].split('.')[0]

            # if name.split('/')[-1] not in test_16task_names:
            #     continue
            test_img.append(args.data_root_path + line.strip().split()[0])
            test_lbl.append(args.data_root_path + line.strip().split()[1])
            test_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    if args.phase == 'train':
        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)

        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler
    
    
    if args.phase == 'validation':
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=list_data_collate)
        return val_loader, val_transforms
    
    
    if args.phase == 'test':
        if args.cache_dataset:
            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=list_data_collate)
        return test_loader, val_transforms

def get_loader_folds(args, fold=1, phase='training'):
    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            # LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),  # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []

    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []

    with open(args.data_txt_path + args.dataset_list[0], 'r') as file:
        all_case_dict = json.load(file)['train']

    for pair_img_mask in all_case_dict:
        if fold != pair_img_mask['fold']:
            name = pair_img_mask['image'].replace('.nii.gz', '')#.split('_')[0]
            train_img.append(args.data_root_path + '/' + pair_img_mask['image'])
            train_lbl.append(args.data_root_path + '/' + pair_img_mask['label'])
            train_post_lbl.append(args.data_root_path + '/' + pair_img_mask['label'].replace('label', 'post_label').replace('.nii.gz', '.h5'))
            train_name.append(name)
        if fold == pair_img_mask['fold']:
            name = pair_img_mask['image'].replace('.nii.gz', '')#.split('_')[0]
            val_img.append(args.data_root_path + '/' + pair_img_mask['image'])
            val_lbl.append(args.data_root_path + '/' + pair_img_mask['label'])
            val_post_lbl.append(args.data_root_path + '/' + pair_img_mask['label'].replace('label', 'post_label').replace('.nii.gz', '.h5'))
            val_name.append(name)
    # for item in args.dataset_list:
    #     for line in open(args.data_txt_path + item + f'_fold{fold}.txt'):
    #         name = line.strip().split()[1].split('.')[0]
    #         train_img.append(args.data_root_path + line.strip().split()[0])
    #         train_lbl.append(args.data_root_path + line.strip().split()[1])
    #         train_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
    #         train_name.append(name)


    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                        for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                        for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('train len {}'.format(len(data_dicts_train)))

    ## validation dict part
    # val_img = []
    # val_lbl = []
    # val_post_lbl = []
    # val_name = []
    # for item in args.dataset_list:
    #     for line in open(args.data_txt_path + item + '_val.txt'):
    #
    #         if '15_3DIRCADb' in line:
    #             name = line.strip().split()[1][:-7]
    #             print(name)
    #         else:
    #             name = line.strip().split()[1].split('.')[0]
    #         val_img.append(args.data_root_path + line.strip().split()[0])
    #         val_lbl.append(args.data_root_path + line.strip().split()[1])
    #         val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
    #         val_name.append(name)
    #
    # data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
    #                   for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    # print('val len {}'.format(len(data_dicts_val)))

    ## test dict part
    # test_img = []
    # test_lbl = []
    # test_post_lbl = []
    # test_name = []
    # for item in args.dataset_list:
    #     for line in open(args.data_txt_path + item + '_test.txt'):
    #         if '15_3DIRCADb' in line:
    #             name = line.strip().split()[1][:-7]
    #             print(name)
    #         else:
    #             name = line.strip().split()[1].split('.')[0]
    #
    #         # if name.split('/')[-1] not in test_16task_names:
    #         #     continue
    #         test_img.append(args.data_root_path + line.strip().split()[0])
    #         test_lbl.append(args.data_root_path + line.strip().split()[1])
    #         test_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
    #         test_name.append(name)
    # data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
    #                    for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    # print('test len {}'.format(len(data_dicts_test)))
    if phase == 'training':
        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms,
                                                    cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms,
                                             cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms,
                                               datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)

        # train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True,
        #                                    shuffle=True) if args.dist else None
        train_sampler = DistributedSampler(dataset=train_dataset) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                  num_workers=args.num_workers,
                                  collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler

    elif phase == 'validation':
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms


    # if args.phase == 'validation':
    #     if args.cache_dataset:
    #         val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
    #     else:
    #         val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
    #     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    #     return val_loader, val_transforms
    #
    # if args.phase == 'test':
    #     if args.cache_dataset:
    #         test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
    #     else:
    #         test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    #     return test_loader, val_transforms

def get_loader_test_folds(args):
    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            # LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),  # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []

    target_dataset = "01_Multi-Atlas_Labeling"

    with open(args.data_txt_path + args.dataset_list[0], 'r') as file:
        all_case_dict = json.load(file)['test']

    print(all_case_dict)
    for pair_img_mask in all_case_dict:
        name = pair_img_mask['image'].replace('.nii.gz', '')  # .split('_')[0]
        val_img.append(args.data_root_path + '/' + pair_img_mask['image'])
        val_lbl.append(args.data_root_path + '/' + pair_img_mask['label'])
        val_post_lbl.append(
            args.data_root_path + '/' + pair_img_mask['label'].replace('label', 'post_label').replace('.nii.gz', '.h5'))

        # name = f"{target_dataset}/img/{pair_img_mask.replace('.nii.gz', '')}" #.split('_')[0]
        # val_img.append(args.data_root_path + '/' + f"{target_dataset}/img/" + pair_img_mask.replace('imagesTs', ''))
        # val_lbl.append(args.data_root_path + '/' + f"{target_dataset}/label/" + pair_img_mask.replace('imagesTs', '').replace('img', 'label'))
        # val_post_lbl.append(args.data_root_path + '/' + f"{target_dataset}/post_label/" + pair_img_mask.replace('imagesTs', '').replace('img', 'post_label').replace('.nii.gz', '.h5'))
        val_name.append(name)

    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                        for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('test len {}'.format(len(data_dicts_val)))

    ## validation dict part
    # val_img = []
    # val_lbl = []
    # val_post_lbl = []
    # val_name = []
    # for item in args.dataset_list:
    #     for line in open(args.data_txt_path + item + '_val.txt'):
    #
    #         if '15_3DIRCADb' in line:
    #             name = line.strip().split()[1][:-7]
    #             print(name)
    #         else:
    #             name = line.strip().split()[1].split('.')[0]
    #         val_img.append(args.data_root_path + line.strip().split()[0])
    #         val_lbl.append(args.data_root_path + line.strip().split()[1])
    #         val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
    #         val_name.append(name)
    #
    # data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
    #                   for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    # print('val len {}'.format(len(data_dicts_val)))

    ## test dict part
    # test_img = []
    # test_lbl = []
    # test_post_lbl = []
    # test_name = []
    # for item in args.dataset_list:
    #     for line in open(args.data_txt_path + item + '_test.txt'):
    #         if '15_3DIRCADb' in line:
    #             name = line.strip().split()[1][:-7]
    #             print(name)
    #         else:
    #             name = line.strip().split()[1].split('.')[0]
    #
    #         # if name.split('/')[-1] not in test_16task_names:
    #         #     continue
    #         test_img.append(args.data_root_path + line.strip().split()[0])
    #         test_lbl.append(args.data_root_path + line.strip().split()[1])
    #         test_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
    #         test_name.append(name)
    # data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
    #                    for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    # print('test len {}'.format(len(data_dicts_test)))

    if args.cache_dataset:
        val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
    else:
        val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    return val_loader, val_transforms


    # if args.phase == 'validation':
    #     if args.cache_dataset:
    #         val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
    #     else:
    #         val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
    #     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    #     return val_loader, val_transforms
    #
    # if args.phase == 'test':
    #     if args.cache_dataset:
    #         test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
    #     else:
    #         test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    #     return test_loader, val_transforms

def get_loader_orignal(args):
    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            # Lambda(func=lambda x: x[0] if isinstance(x, list) else x),  # 解包列表
            # Lambda(func=lambda x: torch.cat(x, dim=0) if isinstance(x, list) else x),  # 解包列表
            
            # CastToTyped(keys=["image"], dtype=torch.float32),
            # CastToTyped(keys=["label", "post_label"], dtype=torch.int8),
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []

    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_train.txt'):
            name = line.strip().split()[1].split('.')[0]
            train_img.append(args.data_root_path + line.strip().split()[0])
            train_lbl.append(args.data_root_path + line.strip().split()[1])
            train_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            train_name.append(name)
    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))


    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_val.txt'):

            if '15_3DIRCADb' in line:
                name = line.strip().split()[1][:-7]
                print(name)
            else:
                name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)

    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))


    ## test dict part
    test_img = []
    test_lbl = []
    test_post_lbl = []
    test_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_test.txt'):
            if '15_3DIRCADb' in line:
                name = line.strip().split()[1][:-7]
                print(name)
            else:
                name = line.strip().split()[1].split('.')[0]

            # if name.split('/')[-1] not in test_16task_names:
            #     continue
            test_img.append(args.data_root_path + line.strip().split()[0])
            test_lbl.append(args.data_root_path + line.strip().split()[1])
            test_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    if args.phase == 'train':
        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)

        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler
    
    
    if args.phase == 'validation':
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=list_data_collate)
        return val_loader, val_transforms
    
    
    if args.phase == 'test':
        if args.cache_dataset:
            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=list_data_collate)
        return test_loader, val_transforms


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='unet', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--network_name', default='Universal_model', choices=['Universal_model', 'UniSeg_model', 'UniSeg_model_first_step'])
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding',
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth',
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=200, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=5, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['BTCV_folds_new.json']) # 'PAOT', 'felix', 'PAOT_10_inner'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='/home/ubuntu//T/CVPR2024/UniversalModel/Datasets/', help='data root path')
    parser.add_argument('--data_txt_path', default='./', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05',
                                            '07', '08', '09', '12', '13', '10_03',
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    # parser.add_argument('--datasetkey', nargs='+', default=['01'],
    #                                         help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--word_encoding', type=str, default='word_embedding', choices=['rand_embedding', 'non', 'word_embedding'])
    parser.add_argument('--word_embedding_path', type=str, default='./expl icit_prompt.pt', help='Name of Experiment')

    args = parser.parse_args()

    train_loader, _ = get_loader_folds(args, fold=0, phase='training')
    for index, item in enumerate(train_loader):
        img1, label, post_label = item['image'][0], item['label'][0], item['post_label'][0]
        break

    nonzero_index = []
    for index, x in enumerate(post_label.reshape(32, -1)):
        if x.max() == 1: nonzero_index.append(index)

    print(img1.shape, post_label.shape, nonzero_index, post_label[1].sum(), post_label[5].sum(), post_label[6].sum(), item['name'])
    f, ax = plt.subplots(1, 6)
    ax[0].imshow(img1[0, :, :, 48])
    ax[1].imshow(post_label[0, :, :, 48])
    ax[2].imshow(post_label[2, :, :, 48])
    ax[3].imshow(post_label[1, :, :, 48])
    ax[4].imshow(post_label[5, :, :, 48])
    ax[5].imshow(post_label[6, :, :, 48])
    plt.show()