"""BigEarthNetv2 dataset."""

import glob
import os
from typing import Callable, Optional
from collections.abc import Sequence

import kornia.augmentation as K
import pandas as pd
import rasterio
import torch
from torch import Generator, Tensor
from torch.utils.data import random_split
from torchgeo.datasets import BigEarthNet

from pyproj import Transformer
from datetime import date
import numpy as np
import pdb


class BigEarthNetv2(BigEarthNet):
    """BigEarthNetv2 dataset.

    Automatic download not implemented, get data from below link.
    """

    splits_metadata = {
        "train": {
            #"url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata-5%.parquet", #"metadata-10%.parquet",
        },
        "val": {
            #"url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata-5%.parquet", #"metadata-10%.parquet",
        },
        "test": {
            #"url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata-5%.parquet", #"metadata-10%.parquet",
        },
    }
    metadata_locs = {
        "s1": {
            #"url": "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst",
            "md5": "",  # unknown
            #"filename": "BigEarthNet-S1.tar.zst",
            "directory": "BigEarthNet-S1-10%",
        },
        "s2": {
            #"url": "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst",
            "md5": "",  # unknown
            #"filename": "BigEarthNet-S2.tar.zst",
            "directory": "BigEarthNet-S2-10%",
        },
        "maps": {
            #"url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "",  # unknown
            #"filename": "Reference_Maps.zst",
            "directory": "Reference_Maps-10%",
        },
    }
    image_size = (120, 120)
    all_band_names_s1 = ('VV','VH')
    all_band_names_s2 = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        band_names: Sequence[str] = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(
            root=root,
            split=split,
            bands=bands,
            num_classes=num_classes,
            transforms=transforms,
            download=download,
            checksum=checksum,
        )

        self.band_names = band_names
        if self.bands == 's1':
            self.all_band_names = self.all_band_names_s1
        else:
            self.all_band_names = self.all_band_names_s2
        self.band_indices = [(self.all_band_names.index(b)+1) for b in band_names if b in self.all_band_names]
        #pdb.set_trace()


        self.class2idx_43 = {c: i for i, c in enumerate(self.class_sets[43])}
        self.class2idx_19 = {c: i for i, c in enumerate(self.class_sets[19])}
        # self._verify()
        self.folders = self._load_folders()

        self.patch_area = (16*10/1000)**2
        self.reference_date = date(1970, 1, 1)

    def get_class2idx(self, label: str, level=19):
        assert level == 19 or level == 43, "level must be 19 or 43"
        return self.class2idx_19[label] if level == 19 else self.class2idx_43[label]

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        pass

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata_locs["s1"]["directory"]
        dir_s2 = self.metadata_locs["s2"]["directory"]
        dir_maps = self.metadata_locs["maps"]["directory"]

        self.metadata = pd.read_parquet(os.path.join(self.root, filename))

        def construct_folder_path(root, dir, patch_id, remove_last: int = 2):
            tile_id = "_".join(patch_id.split("_")[:-remove_last])
            return os.path.join(root, dir, tile_id, patch_id)

        folders = [
            {
                "s1": construct_folder_path(self.root, dir_s1, row["s1_name"], 3),
                "s2": construct_folder_path(self.root, dir_s2, row["patch_id"], 2),
                "maps": construct_folder_path(self.root, dir_maps, row["patch_id"], 2),
            }
            for _, row in self.metadata.iterrows()
        ]

        return folders

    def _load_map_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder_maps = self.folders[index]["maps"]
        paths_maps = glob.glob(os.path.join(folder_maps, "*_reference_map.tif"))
        paths_maps = sorted(paths_maps)
        return paths_maps

    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_map_paths(index)
        map = None
        for path in paths:
            with rasterio.open(path) as dataset:
                map = dataset.read(
                    # indexes=1,
                    # out_shape=self.image_size,
                    out_dtype="int32",
                    # resampling=Resampling.bilinear,
                )
        return torch.from_numpy(map).float()

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """

        image_labels = self.metadata.iloc[index]["labels"]

        # labels -> indices
        indices = [
            self.get_class2idx(label, level=self.num_classes) for label in image_labels
        ]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1

        return image_target

    def _load_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        # if self.bands == 'all':
        #     folder_s1 = self.folders[index]['s1']
        #     folder_s2 = self.folders[index]['s2']
        #     paths_s1 = glob.glob(os.path.join(folder_s1, '*.tif'))
        #     paths_s2 = glob.glob(os.path.join(folder_s2, '*.tif'))
        #     paths_s1 = sorted(paths_s1)
        #     paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
        #     paths = paths_s1 + paths_s2
        if self.bands == 's1':
            folder = self.folders[index]['s1']
            #paths = glob.glob(os.path.join(folder, '*.tif'))
            paths = glob.glob(os.path.join(folder, '*_allbands.tif'))
            paths = sorted(paths)
        else:
            folder = self.folders[index]['s2']
            #paths = glob.glob(os.path.join(folder, '*.tif'))
            paths = glob.glob(os.path.join(folder, '*_allbands.tif'))
            #paths = sorted(paths, key=sort_sentinel2_bands)
            paths = sorted(paths)

        return paths

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        #images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as src:
                array = src.read(
                    self.band_indices,
                ).astype('float32')

                cx,cy = src.xy(src.height // 2, src.width // 2)
                if src.crs.to_string() != 'EPSG:4326':
                    crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                    lon, lat = crs_transformer.transform(cx,cy)
                else:
                    lon, lat = cx, cy

                #pdb.set_trace()
                if self.bands == 's1':
                    date_str = path.split('/')[-1].split('_')[4]
                else:
                    date_str = path.split('/')[-1].split('_')[2]
                date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                delta = (date_obj - self.reference_date).days

                #images.append(array)
        #arrays: np.typing.NDArray[np.int_] = np.stack(images, axis=0)
        #arrays: np.typing.NDArray[np.int_] = np.concatenate(images, axis=0)

        tensor = torch.from_numpy(array).float()
        return tensor, (lon,lat), delta

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        #pdb.set_trace()
        image, coord, delta = self._load_image(index)
        meta_info = np.array([coord[0], coord[1], delta, self.patch_area]).astype(np.float32)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {'image': image, 'label': label, 'meta':meta_info}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample



class ClsDataAugmentation(torch.nn.Module):

    def __init__(self, split, size, bands, band_stats):
        super().__init__()

        self.bands = bands

        if band_stats is not None:
            mean = band_stats['mean']
            std = band_stats['std']
        else:
            mean = [0.0]
            std = [1.0]

        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple."""
        if self.bands == "rgb":
            sample["image"] = sample["image"][1:4, ...].flip(dims=(0,))
            # get in rgb order and then normalization can be applied
        x_out = self.transform(sample["image"]).squeeze(0)
        return x_out, sample["label"]


class SenBenchBenV2Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.modality
        self.band_names = config.band_names
        self.num_classes = config.num_classes
        self.band_stats = config.band_stats

        if self.bands == "rgb":
            # start with rgb and extract later
            self.input_bands = "s2"
        else:
            self.input_bands = self.bands

    def create_dataset(self):
        train_transform = ClsDataAugmentation(
            split="train", size=self.img_size, bands=self.bands, band_stats=self.band_stats
        )
        eval_transform = ClsDataAugmentation(
            split="test", size=self.img_size, bands=self.bands, band_stats=self.band_stats
        )

        dataset_train = BigEarthNetv2(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="train",
            bands=self.input_bands,
            band_names=self.band_names,
            transforms=train_transform,
        )

        # num_subset_samples = int(0.1 * len(dataset_train))
        # # Split the dataset into the subset and the remaining part
        # subset_train, _ = random_split(
        #     dataset_train,
        #     [num_subset_samples, len(dataset_train) - num_subset_samples],
        #     generator=Generator().manual_seed(42),
        # )

        dataset_val = BigEarthNetv2(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="val",
            bands=self.input_bands,
            band_names=self.band_names,
            transforms=eval_transform,
        )
        dataset_test = BigEarthNetv2(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="test",
            bands=self.input_bands,
            band_names=self.band_names,
            transforms=eval_transform,
        )

        return dataset_train, dataset_val, dataset_test
