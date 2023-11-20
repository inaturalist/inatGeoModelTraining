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
        "export_dir": "/Users/alex/Desktop/Machine Learning Development/vision-export-20221020154449-Galliformes-573",
        "export_short_version": "galliformes",
        "train_only_cid_data": True,
        "train_only_wild_data": False,
        "h3_resolution": 6,
        "num_random_samples": 100_000,
        "elevation_file": "/Users/alex/Desktop/Machine Learning Development/elevation_h3_resolution6.csv",
        "experiment_dir": "/Users/alex/Desktop/Machine Learning Development/geo_model_macos/2_8_macos",
        "batch_size": 1024,
        "num_epochs": 10,
        "initial_lr": 0.0005,
        "shuffle_buffer_size": 50_000,
        "full_shuffle_before_tfrecords": False,
    }

    ds = DiscretizedInatGeoModelDataset(config=params)
    ds.make_dataset()

    trainer = DiscretizedInatGeoModelTrainer(config=params)
    trainer.train_geomodel()


if __name__ == "__main__":
    main()
