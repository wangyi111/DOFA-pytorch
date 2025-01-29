import kornia as K
import torch
from torchgeo.datasets import CloudCoverDetection
from typing import ClassVar, TypeAlias
from collections.abc import Callable, Sequence
from torch import Tensor
from datetime import date
import os
import pandas as pd
import numpy as np
import rasterio
from pyproj import Transformer

Path: TypeAlias = str | os.PathLike[str]

class SenBenchCloudS2(CloudCoverDetection):
    url = None
    all_bands = ('B02', 'B03', 'B04', 'B08')
    splits: ClassVar[dict[str, str]] = {'train': 'public', 'val': 'private', 'test': 'private'}

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        #super().__init__(root=root, split=split, bands=bands, transforms=transforms, download=download)
        assert split in self.splits
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self.csv = os.path.join(self.root, self.split, f'{self.split}_metadata.csv')
        self._verify()

        self.metadata = pd.read_csv(self.csv)
        
        self.reference_date = date(1970, 1, 1)
        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Returns a sample from dataset.

        Args:
            index: index to return

        Returns:
            data, metadata (lon,lat,days,area) and label at given index
        """
        chip_id = self.metadata.iat[index, 0]
        date_str = self.metadata.iat[index, 2]
        date_obj = date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]))
        delta = (date_obj - self.reference_date).days

        image, coord = self._load_image(chip_id)
        label = self._load_target(chip_id)

        meta_info = np.array([coord[0], coord[1], delta, self.patch_area]).astype(np.float32)

        sample = {'image': image, 'mask': label, 'meta': torch.from_numpy(meta_info)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        # # add metadata
        # sample['meta'] = torch.from_numpy(meta_info)

        return sample

    def _load_image(self, chip_id: str) -> Tensor:
        """Load all source images for a chip.

        Args:
            chip_id: ID of the chip.

        Returns:
            a tensor of stacked source image data, coord (lon,lat)
        """
        path = os.path.join(self.root, self.split, f'{self.split}_features', chip_id)
        images = []
        coords = None
        for band in self.bands:
            with rasterio.open(os.path.join(path, f'{band}.tif')) as src:
                images.append(src.read(1).astype(np.float32))
                if coords is None:
                    cx,cy = src.xy(src.height // 2, src.width // 2)
                    if src.crs.to_string() != 'EPSG:4326':
                        crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                        lon, lat = crs_transformer.transform(cx,cy)
                    else:
                        lon, lat = cx, cy

        return torch.from_numpy(np.stack(images, axis=0)), (lon,lat)



class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        super().__init__()

        mean = torch.Tensor([0.0])
        std = torch.Tensor([1.0])

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

class SenBenchCloudS2Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size)

        dataset_train = SenBenchCloudS2(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = SenBenchCloudS2(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = SenBenchCloudS2(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test