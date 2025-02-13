import kornia.augmentation as K
import torch
from torchgeo.datasets import EuroSAT
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
from pyproj import Transformer
from typing import TypeAlias, ClassVar
import pathlib
Path: TypeAlias = str | os.PathLike[str]

class SenBenchEuroSATS1(EuroSAT):
    url = None
    base_dir = 'all_imgs'
    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'train': 'eurosat-train.txt',
        'val': 'eurosat-val.txt',
        'test': 'eurosat-test.txt',
    }
    all_band_names = ('VV','VH')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = ['VV','VH'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        self._verify()

        self.valid_fns = []
        self.classes = []
        with open(os.path.join(self.root, self.split_filenames[split])) as f:
            for fn in f:
                self.valid_fns.append(fn.strip().replace('.jpg', '.tif'))
                cls_name = fn.strip().split('_')[0]
                if cls_name not in self.classes:
                    self.classes.append(cls_name)
        self.classes = sorted(self.classes)

        self.root = os.path.join(self.root, self.base_dir)
        #root_path = pathlib.Path(root,split)
        #self.classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m

    def __len__(self):
        return len(self.valid_fns)

    def __getitem__(self, index):

        image, coord, label = self._load_image(index)
        meta_info = np.array([coord[0], coord[1], np.nan, self.patch_area]).astype(np.float32)
        sample = {'image': image, 'label': label, 'meta': torch.from_numpy(meta_info)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        fname = self.valid_fns[index]
        dirname = fname.split('_')[0]
        img_path = os.path.join(self.root, dirname, fname)
        target = self.class_to_idx[dirname]

        with rasterio.open(img_path) as src:
            image = src.read(self.band_indices).astype('float32')
            cx,cy = src.xy(src.height // 2, src.width // 2)
            if src.crs.to_string() != 'EPSG:4326':
                crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                lon, lat = crs_transformer.transform(cx,cy)
            else:
                lon, lat = cx, cy

        return torch.from_numpy(image), (lon,lat), target


class ClsDataAugmentation(torch.nn.Module):
    BAND_STATS = {
        'mean': {
            # 'B01': 1353.72696296,
            # 'B02': 1117.20222222,
            # 'B03': 1041.8842963,
            # 'B04': 946.554,
            # 'B05': 1199.18896296,
            # 'B06': 2003.00696296,
            # 'B07': 2374.00874074,
            # 'B08': 2301.22014815,
            # 'B8A': 2599.78311111,
            # 'B09': 732.18207407,
            # 'B10': 12.09952894,
            # 'B11': 1820.69659259,
            # 'B12': 1118.20259259,
            'VV': -12.54847273,
            'VH': -20.19237134
        },
        'std': {
            # 'B01': 897.27143653,
            # 'B02': 736.01759721,
            # 'B03': 684.77615743,
            # 'B04': 620.02902871,
            # 'B05': 791.86263829,
            # 'B06': 1341.28018273,
            # 'B07': 1595.39989386,
            # 'B08': 1545.52915718,
            # 'B8A': 1750.12066835,
            # 'B09': 475.11595216,
            # 'B10': 98.26600935,
            # 'B11': 1216.48651476,
            # 'B12': 736.6981037,
            'VV': 5.25697717,
            'VH': 5.91150917
        }
    }

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


class SenBenchEuroSATS1Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.band_stats = config.band_stats

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = SenBenchEuroSATS1(
            root=self.root_dir, split="train", bands=self.bands, transforms=train_transform
        )
        dataset_val = SenBenchEuroSATS1(
            root=self.root_dir, split="val", bands=self.bands, transforms=eval_transform
        )
        dataset_test = SenBenchEuroSATS1(
            root=self.root_dir, split="test", bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test