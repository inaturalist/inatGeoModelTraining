import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import glob
import h3pandas
import json
import math
import numpy as np
import os
import pandas as pd
import pathlib
import pendulum
import tensorflow as tf
import time
import uuid

from geo_model_trainer import DiscretizedInatGeoModelTrainer
from geo_dataset_maker import DiscretizedInatGeoModelDataset

params = {
    "export_dir": "/data-ssd/alex/datasets/vision-export-20221020154449-Galliformes-573",
    "export_short_version": "galliformes",
    "train_only_cid_data": True,
    "train_only_wild_data": False,
    "h3_resolution": 6,
    "num_random_samples": 100_000,
    "elevation_file": "/data-ssd/alex/datasets/elevation_h3_resolution6.csv",
    "experiment_dir": "/data-ssd/alex/experiments/geo_prior_tf/galliformes/",
    "batch_size": 1024,
    "num_epochs": 200,
    "initial_lr": 0.0005,
    "shuffle_buffer_size": 10_000,
    "full_shuffle_before_tfrecords": True,
}

dag = DAG(
    dag_id="train_geo_model",
    start_date=pendulum.today("UTC"),
    schedule=None,
    params=params,
)


def _make_training_dataset(**context):
    ds = DiscretizedInatGeoModelDataset(config=context)
    ds.make_dataset()


make_training_dataset = PythonOperator(
    task_id="make_training_dataset", python_callable=_make_training_dataset, dag=dag
)


def _train_geomodel(**context):
    trainer = DiscretizedInatGeoModelTrainer(config=context)
    trainer.train_geomodel()


train_geomodel = PythonOperator(
    task_id="train_geomodel", python_callable=_train_geomodel, dag=dag
)

make_training_dataset >> train_geomodel
