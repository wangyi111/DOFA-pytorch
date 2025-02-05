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

import logging

logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

class SenBenchDFC2020(NonGeoDataset):
    url = None
    #base_dir = 'all_imgs'
    splits = ('train', 'val', 'test')

    label_filenames = {
        'train': 'dfc-train.csv',
        'val': 'dfc-val.csv',
        'test': 'dfc-test.csv',
    }
    s1_band_names = (
        'VV', 'VH'
    )
    s2_band_names = (
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
    )

    rgb_band_names = ('B04', 'B03', 'B02')

    Cls_index = {
        'Background': 0, # to be ignored
        'Forest': 1,
        'Shrubland': 2,
        'Savanna': 3,
        'Grassland': 4,
        'Wetland': 5,
        'Cropland': 6,
        'Urban/Built-up': 7,
        'Snow/Ice': 8,
        'Barren': 9,
        'Water': 10
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = s2_band_names,
        modality = 's2',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self.bands = bands
        self.modality = modality
        if self.modality== 's1':
            self.all_band_names = self.s1_band_names
        else:
            self.all_band_names = self.s2_band_names
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        self.img_dir = os.path.join(self.root, modality)
        self.label_dir = os.path.join(self.root, 'dfc')
        
        self.label_csv = os.path.join(self.root, self.label_filenames[split])
        self.label_fnames = []
        with open(self.label_csv, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname = line.strip()
                self.label_fnames.append(fname)

        #self.reference_date = date(1970, 1, 1)
        self.patch_area = (16*10/1000)**2 # patchsize 8 pix, gsd 300m

    def __len__(self):
        return len(self.label_fnames)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        label = self._load_target(index)
        sample = {'image': images, 'mask': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        label_fname = self.label_fnames[index]
        img_fname = label_fname.replace('dfc',self.modality)
        img_path = os.path.join(self.img_dir, img_fname)
        
        with rasterio.open(img_path) as src:
            img = src.read(self.band_indices).astype('float32')
            img = torch.from_numpy(img)

            # # get lon, lat
            # cx,cy = src.xy(src.height // 2, src.width // 2)
            # if src.crs.to_string() != 'EPSG:4326':
            #     # convert to lon, lat
            #     crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
            #     lon, lat = crs_transformer.transform(cx,cy)
            # else:
            #     lon, lat = cx, cy
            # # get time
            # img_fname = os.path.basename(s3_path)
            # date_str = img_fname.split('____')[1][:8]
            # date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            # delta = (date_obj - self.reference_date).days
            meta_info = np.array([np.nan, np.nan, np.nan, self.patch_area]).astype(np.float32)
            meta_info = torch.from_numpy(meta_info)

        return img, meta_info

    def _load_target(self, index):

        label_fname = self.label_fnames[index]
        label_path = os.path.join(self.label_dir, label_fname)

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label[label==0] = 256
            label = label - 1
            labels = torch.from_numpy(label).long()

        return labels



class SegDataAugmentation(torch.nn.Module):
    BAND_STATS = {
        'mean': {
            'B01': 1353.72696296,
            'B02': 1117.20222222,
            'B03': 1041.8842963,
            'B04': 946.554,
            'B05': 1199.18896296,
            'B06': 2003.00696296,
            'B07': 2374.00874074,
            'B08': 2301.22014815,
            'B8A': 2599.78311111,
            'B09': 732.18207407,
            'B10': 12.09952894,
            'B11': 1820.69659259,
            'B12': 1118.20259259,
            'VV': -12.54847273,
            'VH': -20.19237134
        },
        'std': {
            'B01': 897.27143653,
            'B02': 736.01759721,
            'B03': 684.77615743,
            'B04': 620.02902871,
            'B05': 791.86263829,
            'B06': 1341.28018273,
            'B07': 1595.39989386,
            'B08': 1545.52915718,
            'B8A': 1750.12066835,
            'B09': 475.11595216,
            'B10': 98.26600935,
            'B11': 1216.48651476,
            'B12': 736.6981037,
            'VV': 5.25697717,
            'VH': 5.91150917
        }
    }
    def __init__(self, split, size, bands):
        super().__init__()

        mean = []
        std = []
        for band in bands:
            mean.append(self.BAND_STATS['mean'][band])
            std.append(self.BAND_STATS['std'][band])
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


class SenBenchDFC2020Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.modality = config.modality

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size, bands=self.bands)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size, bands=self.bands)

        dataset_train = SenBenchDFC2020(
            root=self.root_dir, split="train", bands=self.bands, modality=self.modality, transforms=train_transform
        )
        dataset_val = SenBenchDFC2020(
            root=self.root_dir, split="val", bands=self.bands, modality=self.modality, transforms=eval_transform
        )
        dataset_test = SenBenchDFC2020(
            root=self.root_dir, split="test", bands=self.bands, modality=self.modality, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test