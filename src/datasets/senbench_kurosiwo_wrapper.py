import torch
from torchgeo.datasets.geo import NonGeoDataset
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
from pyproj import Transformer
from datetime import date
from typing import TypeAlias, ClassVar
import rioxarray as rio
from torchvision import transforms
import kornia as K
import logging
import pickle
import cv2 as cv
from datetime import datetime


logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

def get_grids(pickle_path):
    if not os.path.isfile(pickle_path):
        print("Pickle file not found! ", pickle_path)
        exit(2)
    with open(pickle_path, "rb") as file:
        grid_dict = pickle.load(file)
    return grid_dict



class SenBenchKuroSiwo(NonGeoDataset):
    url = None
    splits = ('train', 'val', 'test')
    data_mean = [0.0904, 0.0241]
    data_std =  [0.0468, 0.0226]
    dem_mean = 82.96274925580951 
    dem_std = 153.71243439980663
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        train_on_water_samples_only: bool = False,
        train_acts = [
            130,
            470,
            555,
            118,
            174,
            324,
            421,
            554,
            427,
            518,
            502,
            498,
            497,
            496,
            492,
            147,
            267,
            273,
            275,
            417,
            567,1111011,1111004,1111009,1111010,1111006,1111005
        ],
        val_acts = [514, 559, 279, 520, 437,1111003,1111008],
        test_acts = [321, 561, 445, 562, 411, 1111002, 277,1111007,205,1111013],
        clamp_input = 0.15,
        use_dem = False,
        channels = [ "vv","vh"],
    ) -> None:

        self.root = root
        self.data_root = os.path.join(root, "data")
        self.download = download
        self.transforms = transforms
        self.water_samples_metadata = os.path.join(root,'pickles','grid_dict_water.pkl')
        self.all_samples_metadata = os.path.join(root,'pickles','grid_dict_full.pkl')
        self.channels = channels
        self.clamp_input = clamp_input
        self.train_acts = train_acts
        self.val_acts = val_acts
        self.test_acts = test_acts
        self.use_dem = use_dem
        self.patch_area = (224*10/1000)**2 # patchsize 224 pix, gsd 10m
        self.clz_stats = {1: 0, 2: 0, 3: 0}
        self.act_stats = {}
        
        assert split in ['train', 'val', 'test']

        self.mode = split

        if self.mode == "train":
            self.valid_acts = self.train_acts
            if train_on_water_samples_only:
                self.pickle_path = self.water_samples_metadata
            else:
                self.pickle_path = self.all_samples_metadata
        elif self.mode == "val":
            self.valid_acts = self.val_acts
            self.pickle_path = self.all_samples_metadata
        else:
            self.valid_acts = self.test_acts
            self.pickle_path = self.all_samples_metadata
        
        total_grids = {}
        self.positive_records = []
        self.negative_records = []

        self.grids = get_grids(pickle_path=self.pickle_path)        

        all_activations = []
        all_activations.extend(self.train_acts)
        all_activations.extend(self.val_acts)
        all_activations.extend(self.test_acts)
        self.records = []
        for key in self.grids:
            record = {}
            record["id"] = key
            record["path"] = self.grids[key]["path"]

            record["info"] = self.grids[key]["info"]
            record["type"] = None
            record["clz"] = self.grids[key]["clz"]
            activation = self.grids[key]["info"]["actid"]
            aoi = self.grids[key]["info"]["aoiid"]
            act_aoi = activation

            record["activation"] = activation
            if act_aoi in self.valid_acts:
                self.clz_stats[record["clz"]] += 1
                if act_aoi in self.act_stats:
                    self.act_stats[act_aoi] += 1
                else:
                    self.act_stats[act_aoi] = 1
                
                self.records.append(record)
                if key in self.grids:
                    self.positive_records.append(record)
                else:
                    self.negative_records.append(record)

            if act_aoi not in all_activations and act_aoi not in self.non_valids:
                print("Activation: ", activation, " not in Activations")
                self.non_valids.append(act_aoi)

        print("Samples per Climatic zone for mode: ", self.mode)
        print(self.clz_stats)
        print("Samples per Activation for mode: ", self.mode)
        print(self.act_stats)
        self.num_examples = len(self.records)
        self.activations = set([record["activation"] for record in self.records])

    def __len__(self):
        return self.num_examples

    def concat(self, image1, image2):
        image1_exp = np.expand_dims(image1, 0)  # vv
        image2_exp = np.expand_dims(image2, 0)  # vh

        if set(self.channels) == set(["vv", "vh", "vh/vv"]):
            eps = 1e-7
            image = np.vstack((image1_exp, image2_exp, image2_exp / (image1_exp + eps)))  # vv, vh, vh/vv
        elif set(self.channels) == set(["vv", "vh"]):
            image = np.vstack((image1_exp, image2_exp))  # vv, vh
        elif self.channels == ["vh"]:
            image = image2_exp  # vh

        image = torch.from_numpy(image).float()

        if self.clamp_input is not None:
            image = torch.clamp(image, min=0.0, max=self.clamp_input)
            image = torch.nan_to_num(image, self.clamp_input)
        else:
            image = torch.nan_to_num(image, 200)
        return image
    
    def __getitem__(self, index):

        sample = self.records[index]

        path = sample["path"]
        path = os.path.join(self.data_root, path)
        files = os.listdir(path)
        clz = sample["clz"]
        activation = sample["activation"]
        mask = None
        for file in files:
            current_path = str(os.path.join(path, file))
            if "xml" not in file:
                if file.startswith("MK0_MLU") and (sample["type"] is None):
                    # Get mask of flooded/perm water pixels
                    mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MK0_MNA") and (sample["type"] is None):
                    # Get mask of valid pixels
                    valid_mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MS1_IVV") and (sample["type"] not in ["pre1", "pre2"]):
                    # Get master ivv channel
                    flood_vv = rio.open_rasterio(current_path)
                    post_date = file.split("_")[-1][:-4]
                    post_date = datetime.strptime(post_date, "%Y%m%d")    
                   
                    
                    center_x_pixel = flood_vv.shape[2] // 2
                    center_y_pixel = flood_vv.shape[1] // 2
                    lon = flood_vv.x.values[center_x_pixel]
                    lat = flood_vv.y.values[center_y_pixel]    
                    if flood_vv.rio.crs.to_string() != "EPSG:4326":
                        transformer = Transformer.from_crs(flood_vv.rio.crs, "EPSG:4326", always_xy=True)
                        lon, lat = transformer.transform(lon, lat)
                    flood_vv = flood_vv.to_numpy().squeeze()  

                elif file.startswith("MS1_IVH") and (sample["type"] not in ["pre1", "pre2"]):
                    # Get master ivh channel
                    flood_vh = rio.open_rasterio(current_path).to_numpy().squeeze()
                    
                elif file.startswith("SL1_IVV") and (sample["type"] not in ["flood", "pre2"]):
                    # Get slave1 vv channel
                    sec1_vv = rio.open_rasterio(current_path).to_numpy().squeeze()
                    pre1_date = file.split("_")[-1][:-4]
                    pre1_date = datetime.strptime(pre1_date, "%Y%m%d")    
                elif file.startswith("SL1_IVH") and (sample["type"] not in ["flood", "pre2"]):
                    # Get sl1 vh channel
                    sec1_vh = rio.open_rasterio(current_path).to_numpy().squeeze()

                elif file.startswith("SL2_IVV") and (sample["type"] not in ["flood", "pre1"]):
                    # Get sl2 vv channel
                    sec2_vv = rio.open_rasterio(current_path).to_numpy().squeeze()
                    pre2_date = file.split("_")[-1][:-4]
                    pre2_date = datetime.strptime(pre2_date, "%Y%m%d")    
                elif file.startswith("SL2_IVH") and (sample["type"] not in ["flood", "pre1"]):
                    # Get sl2 vh channel
                    sec2_vh = rio.open_rasterio(current_path).to_numpy().squeeze()
                elif file.startswith("MK0_DEM"):
                    if self.use_dem:
                        # Get DEM
                        dem = rio.open_rasterio(current_path)
                        nans = dem.isnull()
                        if nans.any():
                            dem = dem.rio.interpolate_na()
                            nans = dem.isnull()

                        dem = dem.to_numpy()
                                            
                    else:
                        dem = None

        # Concat channels
        if sample["type"] not in ("pre1", "pre2"):
            flood = self.concat(flood_vv, flood_vh)
        if sample["type"] not in ("flood", "pre2"):
            pre_event_1 = self.concat(sec1_vv, sec1_vh)
        if sample["type"] not in ("flood", "pre1"):
            pre_event_2 = self.concat(sec2_vv, sec2_vh)

        if sample["type"] is None:
            if mask is None:
                mask = np.zeros((224, 224))

        mask = torch.from_numpy(mask).long()

        # Return record given training options
        if sample["type"] == "pre1":
            return pre_event_1
        if sample["type"] == "pre2":
            return pre_event_2
        if sample["type"] == "flood":
            return flood

        valid_mask = torch.from_numpy(valid_mask)

        mask = mask.long()

        image = torch.vstack([flood, pre_event_1, pre_event_2])
        if self.use_dem:
            image = torch.vstack([image, torch.from_numpy(dem)]) 
        
        
        #Get avg difference between pre-event 1, pre-event 2 and post date
        pre1_diff = (post_date - pre1_date).days
        pre2_diff = (post_date - pre2_date).days
        avg_diff = (pre1_diff + pre2_diff) / 2
        meta_info = np.array([lon, lat, avg_diff, self.patch_area]).astype(np.float32)
        meta_info = torch.from_numpy(meta_info)
        sample = {"image": image, "groundtruth": mask, "meta": meta_info}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def plot_samples(self,image, mask,savefig_path=None):
        """

        Args:
            image: stacked SAR. Assumes both VV and VH channels. Sequence represents flood, pre_event1, pre_event 2 SAR. Plots first channel for each timestep.
            mask: mask of flooded/perm water pixels
            dem: DEM
            savefig_path: path to save the figure. Optional.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        num_figures = 5 if self.use_dem else 4
        fig, axs = plt.subplots(1, num_figures, figsize=(20, 5))
        
        cmap = ListedColormap(['black', 'lightblue', 'purple','green'])
        flood = image[:2]
        pre_event_1 = image[2:4]
        pre_event_2 = image[4:6]
        if self.use_dem:
            print(image.shape)
            dem = image[6]
        axs[0].set_title("Flood")
        axs[1].set_title("Pre-event 1")
        axs[2].set_title("Pre-event 2")            
        axs[0].imshow(flood[0], cmap="gray")
        axs[1].imshow(pre_event_1[0], cmap="gray")
        axs[2].imshow(pre_event_2[0],cmap="gray")
        if self.use_dem:
            axs[3].imshow(dem.squeeze())
            axs[3].set_title("DEM")
            img = axs[4].imshow(mask, cmap=cmap,vmin=0, vmax=3)
            cbar = fig.colorbar(img, ax=axs[4])
            axs[4].set_title("Mask")
        else:
            img = axs[3].imshow(mask, cmap=cmap)
            axs[3].colorbar()
            axs[3].set_title("Mask")
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['No water', 'Permanent Waters', 'Flood','Out of AOI'])
        if savefig_path is not None:
            plt.savefig(savefig_path)
        plt.show()



class KuroSiwoDataAugmentation(torch.nn.Module):
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
        x,mask = batch["image"], batch["groundtruth"]
        x = self.norm(x)
        x_out, mask_out = self.transform(x, mask)
        return x_out.squeeze(0), mask_out.squeeze(0).squeeze(0), batch["meta"]


class SenBenchKuroSiwoDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.band_stats = config.band_stats

    def create_dataset(self):
        train_transform = KuroSiwoDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
        eval_transform = KuroSiwoDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = SenBenchKuroSiwo(
            root=self.root_dir, split="train", bands=self.bands, transforms=train_transform
        )
        dataset_val = SenBenchKuroSiwo(
            root=self.root_dir, split="val", bands=self.bands, transforms=eval_transform
        )
        dataset_test = SenBenchKuroSiwo(
            root=self.root_dir, split="test", bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test

#Main function
if __name__ == "__main__":
    transforms = KuroSiwoDataAugmentation(split="train", size=(224, 224), band_stats= None)#{"mean":[0.0904, 0.0241]*3 + [ 82.96274925580951 ],"std": [0.0468, 0.0226]*3 + [ 153.71243439980663]})
    d = SenBenchKuroSiwo(root="Your_Path", use_dem=True,transforms=transforms,clamp_input=1.,train_on_water_samples_only=True)
    data, mask, meta = d[0]
    print(data.shape)
    d.plot_samples(image=data, mask=mask,savefig_path="sample.png")
    print(mask.unique())
    print(mask.shape)
    print(meta)