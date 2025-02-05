import kornia.augmentation as K
import torch
from torchgeo.datasets import So2Sat
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
from pyproj import Transformer
import h5py
from typing import TypeAlias, ClassVar
import pathlib
Path: TypeAlias = str | os.PathLike[str]

class SenBenchSo2Sat(So2Sat):

    versions = ('3_culture_10')
    filenames_by_version: ClassVar[dict[str, dict[str, str]]] = {
        # '2': {
        #     'train': 'training.h5',
        #     'validation': 'validation.h5',
        #     'test': 'testing.h5',
        # },
        # '3_random': {'train': 'random/training.h5', 'test': 'random/testing.h5'},
        # '3_block': {'train': 'block/training.h5', 'test': 'block/testing.h5'},
        '3_culture_10': {
            'train': 'culture_10/train-new.h5',
            'val': 'culture_10/val-new.h5',
            'test': 'culture_10/test-new.h5',
        },
    }

    classes = (
        'Compact high rise',
        'Compact mid rise',
        'Compact low rise',
        'Open high rise',
        'Open mid rise',
        'Open low rise',
        'Lightweight low rise',
        'Large low rise',
        'Sparsely built',
        'Heavy industry',
        'Dense trees',
        'Scattered trees',
        'Bush, scrub',
        'Low plants',
        'Bare rock or paved',
        'Bare soil or sand',
        'Water',
    )

    all_s1_band_names = (
        'S1_B1', # VH real
        'S1_B2', # VH imaginary
        'S1_B3', # VV real
        'S1_B4', # VV imaginary
        'S1_B5', # VH intensity
        'S1_B6', # VV intensity
        'S1_B7', # PolSAR covariance matrix off-diagonal real
        'S1_B8', # PolSAR covariance matrix off-diagonal imaginary
    )
    all_s2_band_names = (
        'S2_B02',
        'S2_B03',
        'S2_B04',
        'S2_B05',
        'S2_B06',
        'S2_B07',
        'S2_B08',
        'S2_B8A',
        'S2_B11',
        'S2_B12',
    )
    all_band_names = all_s1_band_names + all_s2_band_names

    rgb_bands = ('S2_B04', 'S2_B03', 'S2_B02')

    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': all_band_names,
        's1': all_s1_band_names,
        's2': all_s2_band_names,
        'rgb': rgb_bands,
    }

    def __init__(
        self,
        root: Path = 'data',
        version: str = '3_culture_10', # only supported version now
        split: str = 'train',
        bands: Sequence[str] = BAND_SETS['s2'], # only supported bands now
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        #h5py = lazy_import('h5py')

        assert version in self.versions
        assert split in self.filenames_by_version[version]

        self._validate_bands(bands)
        self.s1_band_indices: np.typing.NDArray[np.int_] = np.array(
            [
                self.all_s1_band_names.index(b)
                for b in bands
                if b in self.all_s1_band_names
            ]
        ).astype(int)

        self.s1_band_names = [self.all_s1_band_names[i] for i in self.s1_band_indices]

        self.s2_band_indices: np.typing.NDArray[np.int_] = np.array(
            [
                self.all_s2_band_names.index(b)
                for b in bands
                if b in self.all_s2_band_names
            ]
        ).astype(int)

        self.s2_band_names = [self.all_s2_band_names[i] for i in self.s2_band_indices]

        self.bands = bands

        self.root = root
        self.version = version
        self.split = split
        self.transforms = transforms
        # self.checksum = checksum

        self.fn = os.path.join(self.root, self.filenames_by_version[version][split])

        # if not self._check_integrity():
        #     raise DatasetNotFoundError(self)

        with h5py.File(self.fn, 'r') as f:
            self.size: int = f['label'].shape[0]

        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        #h5py = lazy_import('h5py')
        with h5py.File(self.fn, 'r') as f:
            #s1 = f['sen1'][index].astype(np.float32)
            #s1 = np.take(s1, indices=self.s1_band_indices, axis=2)
            s2 = f['sen2'][index].astype(np.float32)
            s2 = np.take(s2, indices=self.s2_band_indices, axis=2)

            # convert one-hot encoding to int64 then torch int
            label = torch.tensor(f['label'][index].argmax())

            #s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
            s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format

            #s1 = torch.from_numpy(s1)
            s2 = torch.from_numpy(s2)

        meta_info = np.array([np.nan, np.nan, np.nan, self.patch_area]).astype(np.float32)

        sample = {'image': s2, 'label': label, 'meta': torch.from_numpy(meta_info)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class ClsDataAugmentation(torch.nn.Module):
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
            #'VV': -12.54847273,
            #'VH': -20.19237134
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
            #'VV': 5.25697717,
            #'VH': 5.91150917
        }
    }

    def __init__(self, split, size, bands):
        super().__init__()

        mean = []
        std = []
        for band in bands:
            band = band[3:]
            mean.append(self.BAND_STATS['mean'][band])
            std.append(self.BAND_STATS['std'][band])
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
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"], batch["meta"]


class SenBenchSo2SatDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.version = config.version

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size, bands=self.bands)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size, bands=self.bands)

        dataset_train = SenBenchSo2Sat(
            root=self.root_dir, version=self.version, split="train", bands=self.bands, transforms=train_transform
        )
        dataset_val = SenBenchSo2Sat(
            root=self.root_dir, version=self.version, split="val", bands=self.bands, transforms=eval_transform
        )
        dataset_test = SenBenchSo2Sat(
            root=self.root_dir, version=self.version, split="test", bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test