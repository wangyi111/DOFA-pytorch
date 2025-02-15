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

import logging
import pickle
import cv2 as cv

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
        channels = [ "vv","vh"]
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download

        self.water_samples_metadata = os.path.join(root,'pickles','grid_dict_water.pkl')
        self.all_samples_metadata = os.path.join(root,'pickles','grid_dict_full.pkl')
        self.channels = channels
        self.clamp_input = clamp_input
        self.train_acts = train_acts
        self.val_acts = val_acts
        self.test_acts = test_acts
        self.use_dem = use_dem
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
        path = os.path.join(self.root_path, path)
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
                    flood_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)                        

                elif file.startswith("MS1_IVH") and (sample["type"] not in ["pre1", "pre2"]):
                    # Get master ivh channel
                    flood_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                    
                elif file.startswith("SL1_IVV") and (sample["type"] not in ["flood", "pre2"]):
                    # Get slave1 vv channel
                    sec1_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL1_IVH") and (sample["type"] not in ["flood", "pre2"]):
                    # Get sl1 vh channel
                    sec1_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL2_IVV") and (sample["type"] not in ["flood", "pre1"]):
                    # Get sl2 vv channel
                    sec2_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH)

                elif file.startswith("SL2_IVH") and (sample["type"] not in ["flood", "pre1"]):
                    # Get sl2 vh channel
                    sec2_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
                elif file.startswith("MK0_DEM"):
                    if self.use_dem:
                        # Get DEM
                        dem = rio.open_rasterio(current_path)
                        nans = dem.isnull()
                        if nans.any():
                            dem = dem.rio.interpolate_na()
                            nans = dem.isnull()

                        nodata = dem.rio.nodata
                        dem = dem.to_numpy()
                        if not self.configs["dem"] and self.configs["slope"]:
                            print("To return the slope the DEM option must be enabled. Validate the config file!")
                            exit(2)

                        
                    
                        dem_normalization = transforms.Normalize(
                                mean=self.configs["dem_mean"],
                                std=self.configs["dem_std"],
                            )
                        dem = dem_normalization(torch.from_numpy(dem))
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

        data_normalization = transforms.Normalize(self.means, self.stds)
        # Scale images if necessary
        if self.configs["scale_input"] is not None:
            valid_mask = valid_mask == 1
            flood = data_normalization(flood)
            pre_event_1 = data_normalization(pre_event_1)
            pre_event_2 = data_normalization(pre_event_2)

        return flood, mask, pre_event_1, pre_event_2, dem


#Main function
if __name__ == "__main__":
    d = SenBenchKuroSiwo(root="")
    print(len(d))