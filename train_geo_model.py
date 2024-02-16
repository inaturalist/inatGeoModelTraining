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
        "export_dir": "/data-ssd/alex/datasets/vision-export-20231231060015-aka-2.11",
        "export_short_version": "2.11",
        "train_only_cid_data": True,
        "train_only_wild_data": False,
        "h3_resolution": 6,
        "num_random_samples": 100_000,
        "elevation_file": "/home/alex/elevation_h3_resolution6.csv",
        "experiment_dir": "/data-ssd/alex/experiments/geo_prior_tf/2_11",
        "batch_size": 1024,
        "num_epochs": 200,
        "initial_lr": 0.0005,
        "shuffle_buffer_size": 50_000,
        "full_shuffle_before_tfrecords": False,
        "lr_warmup_cosine_decay": True,
    }

    ds = DiscretizedInatGeoModelDataset(config=params)
    ds.make_dataset()

    trainer = DiscretizedInatGeoModelTrainer(config=params)
    trainer.train_geomodel()


if __name__ == "__main__":
    main()
