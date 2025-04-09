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

from lib.geo_model_trainer import DiscretizedInatGeoModelTrainer
from lib.geo_dataset_maker import DiscretizedInatGeoModelDataset


def main():
    params = {
<<<<<<< Updated upstream
        "dataset_type": "inat", # inat or sinr
        "dataset_dir": "/disk/mnt/data/exports/vision-export-20241206190125-aka-small-2.0",
        "export_short_version": "small-2",
=======
        "dataset_type": "inat",  # inat or sinr
        "dataset_dir": "/data-ssd/alex/datasets/vision-export-20250119070012-aka-2.20",
        "export_short_version": "2_20_grid",
>>>>>>> Stashed changes
        "train_only_cid_data": True,
        "train_only_wild_data": False,
        "h3_resolution": 6,
        "num_random_samples": 100_000,
        "elevation_file": "/home/alex/elevation_h3_resolution6.csv",
        "experiment_dir": "/data-ssd/alex/experiments/geo_prior_tf/2_20_grid/",
        "batch_size": 1024,
        "num_epochs": 200,
        "initial_lr": 0.0005,
        "shuffle_buffer_size": 50_000,
        "full_shuffle_before_tfrecords": False,
        "lr_warmup_cosine_decay": True,
        "wandb_project": "geomodel_tf",
        "inner_nodes": False,
    }
<<<<<<< Updated upstream

    params["tfrecord_file"] = params["dataset_dir"] + "/" + "geo_spatial_grid_datasets/" + f"r{params['h3_resolution']}_elevation_empty_cells.tf"
=======
    
    if params["inner_nodes"]:
        filename = f"r{params['h3_resolution']}_empty_cells_with_elevation_inner_nodes_duckdb.tf"
    else:
        filename = f"r{params['h3_resolution']}_elevation_empty_cells.tv"

    params["tfrecord_file"] = os.path.join(
        params["dataset_dir"],
        "geo_spatial_grid_datasets",
        filename
    )
>>>>>>> Stashed changes

    ds = DiscretizedInatGeoModelDataset(config=params)
    ds.make_dataset()

    trainer = DiscretizedInatGeoModelTrainer(config=params)
    trainer.train_geomodel()


if __name__ == "__main__":
    main()
