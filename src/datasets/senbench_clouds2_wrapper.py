import kornia as K
import torch
from torchgeo.datasets.geo import NonGeoDataset
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
import cv2
from pyproj import Transformer
from datetime import date
from typing import TypeAlias, ClassVar
import pathlib
from shapely import wkt
import pandas as pd
import tacoreader

import logging
import pdb

logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

class SenBenchCloudS2(NonGeoDataset):
    url = None
    #base_dir = 'all_imgs'
    all_band_names = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')

    split_filenames = {
        'train': 'cloudsen12-l1c-train.taco',
        'val': 'cloudsen12-l1c-val.taco',
        'test': 'cloudsen12-l1c-test.taco',
    }

    Cls_index_multi = {
        'clear': 0,
        'thick cloud': 1,
        'thin cloud': 2,
        'cloud shadow': 3,
    }



    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_band_names,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self.bands = bands
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        taco_file = os.path.join(root,self.split_filenames[split])
        self.dataset = tacoreader.load(taco_file)
        self.cache = {}

        # filter corrupted entries
        count = 0
        count_corrupted = 0
        #pdb.set_trace()
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset.read(i)
                s2l1c = sample.read(0) # str
                target = sample.read(1) # str
                coord = sample['stac:centroid'][0] # str
                time_start = sample['stac:time_start'][0] # str
                self.cache[count] = (s2l1c, target, coord, time_start)
                count += 1
            except Exception as e:
                count_corrupted += 1
        self.length = count
        print(split,count,"valid samples.")

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        #pdb.set_trace()

        # if index not in self.cache:
        #     sample = self.dataset.read(index)
        #     s2l1c = sample.read(0) # str
        #     target = sample.read(1) # str
        #     coord = sample['stac:centroid'][0] # str
        #     time_start = sample['stac:time_start'][0] # str
        #     self.cache[index] = (s2l1c, target, coord, time_start)
        # else:
        #pdb.set_trace()
        s2l1c, target, coord, time_start = self.cache[index]

        # Open the files and load data
        with rasterio.open(s2l1c) as src, rasterio.open(target) as dst:
            s2l1c_data = src.read(self.band_indices).astype('float32')
            target_data = dst.read(1)
        image = torch.from_numpy(s2l1c_data)
        label = torch.from_numpy(target_data).long()
        
        coord = wkt.loads(coord).coords[0]
        date_obj = pd.to_datetime(time_start, unit='s').date()
        delta = (date_obj - self.reference_date).days
        meta_info = np.array([coord[0], coord[1], delta, self.patch_area]).astype(np.float32)
        meta_info = torch.from_numpy(meta_info)

        sample = {'image': image, 'mask': label, 'meta': meta_info}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size, band_stats):
        super().__init__()

        if band_stats is not None:
            mean = band_stats['mean']
            std = band_stats['std']
        else:
            mean = [0.0]
            std = [1.0]

        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        self.norm = K.augmentation.Normalize(mean=mean, std=std)

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=True),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x,mask = batch["image"], batch["mask"]
        x = self.norm(x)
        x_out, mask_out = self.transform(x, mask)
        return x_out.squeeze(0), mask_out.squeeze(0).squeeze(0), batch["meta"]


class SegDataAugmentationSoftCon(torch.nn.Module):

    def __init__(self, split, size, band_stats):
        super().__init__()

        if band_stats is not None:
            self.mean = band_stats['mean']
            self.std = band_stats['std']
        else:
            self.mean = [0.0]
            self.std = [1.0]

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                #K.augmentation.RandomResizedCrop(size=size, scale=(0.8,1.0)),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=True),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        sample_img,mask = sample["image"], sample["mask"]

        img_bands = []
        for b in range(13):
            img = sample_img[b,:,:].clone()
            ## normalize
            img = self.normalize(img,self.mean[b],self.std[b])         
            img_bands.append(img)
        sample_img = torch.stack(img_bands,dim=0)

        x_out, mask_out = self.transform(sample_img, mask)
        return x_out.squeeze(0), mask_out.squeeze(0).squeeze(0), sample["meta"]

    @torch.no_grad()
    def normalize(self, img, mean, std):
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        return img



class SenBenchCloudS2Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.band_stats = config.band_stats
        self.norm_form = config.norm_form if 'norm_form' in config else None

    def create_dataset(self):
        if self.norm_form == 'softcon':
            train_transform = SegDataAugmentationSoftCon(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = SegDataAugmentationSoftCon(split="test", size=self.img_size, band_stats=self.band_stats)
        else:
            train_transform = SegDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = SegDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = SenBenchCloudS2(
            root=self.root_dir, split="train", bands=self.bands, transforms=train_transform
        )
        dataset_val = SenBenchCloudS2(
            root=self.root_dir, split="val", bands=self.bands, transforms=eval_transform
        )
        dataset_test = SenBenchCloudS2(
            root=self.root_dir, split="test", bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test