import glob
import h3pandas
import json
import math
import numpy as np
import os
import pandas as pd
import pathlib
import tensorflow as tf
import time
import uuid

import click
import yaml

from lib.geo_model_trainer import DiscretizedInatGeoModelTrainer
from lib.geo_dataset_maker import DiscretizedInatGeoModelDataset


@click.command()
@click.option("--config_file", required=True)
def main(config_file):
    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    if params["inner_nodes"]:
        filename = f"r{params['h3_resolution']}_empty_cells_with_elevation_inner_nodes_duckdb.tf"
    else:
        filename = f"r{params['h3_resolution']}_elevation_empty_cells.tv"

    params["tfrecord_file"] = os.path.join(
        params["dataset_dir"],
        "geo_spatial_grid_datasets",
        filename
    )

    ds = DiscretizedInatGeoModelDataset(config=params)
    ds.make_dataset()

    ds.spatial_train = None
    ds.spatial_train_h3_dense = None
    ds.negatives_h3 = None
    ds.combined = None
    ds.combined_with_elevation = None
    ds.spatial_train_h3 = None


    trainer = DiscretizedInatGeoModelTrainer(config=params)
    trainer.train_geomodel()


if __name__ == "__main__":
    main()
