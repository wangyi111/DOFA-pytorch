"""Factory utily functions to create datasets and models."""

from foundation_models import (
    CromaModel,
    ScaleMAEModel,
    GFMModel,
    DinoV2Model,
    SoftConModel,
    DofaModel,
    SatMAEModel,
    AnySatModel,
    DofaSModel,
)
from datasets.geobench_wrapper import GeoBenchDataset
from datasets.resisc_wrapper import Resics45Dataset
from datasets.benv2_wrapper import BenV2Dataset
from datasets.senbench_clouds2_wrapper import SenBenchCloudS2Dataset
from datasets.senbench_eurosats2_wrapper import SenBenchEuroSATS2Dataset
from datasets.senbench_eurosats1_wrapper import SenBenchEuroSATS1Dataset


model_registry = {
    "croma": CromaModel,
    # "panopticon": PanopticonModel,
    "scalemae": ScaleMAEModel,
    "gfm": GFMModel,
    "dinov2": DinoV2Model,
    "softcon": SoftConModel,
    "dofa": DofaModel,
    "satmae": SatMAEModel,
    "anysat": AnySatModel,
    # Add other model mappings here
    "dofas": DofaSModel,
}

dataset_registry = {
    "geobench": GeoBenchDataset,
    "resisc45": Resics45Dataset,
    "benv2": BenV2Dataset,
    # Add other dataset mappings here
    "senbench_clouds2": SenBenchCloudS2Dataset,
    "senbench_eurosats2": SenBenchEuroSATS2Dataset,
    "senbench_eurosats1": SenBenchEuroSATS1Dataset,
}


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")
    dataset = dataset_class(config_data)
    # return the train, val, and test dataset
    return dataset.create_dataset()


def create_model(args, config_model, dataset_config=None):
    model_name = config_model.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    model = model_class(args, config_model, dataset_config)

    return model
