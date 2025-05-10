import json
import math
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import tqdm
import wandb

from lib.models.geo_model_net import make_geo_model_net
from utils import make_rand_samples_tf
from lib.models.encoders import CoordEncoder
from sinr_loss import sinr_loss
from lib.data.data_loader import (
    make_subsampled_dataset,
    load_inat_dataset_from_parquet,
    load_inat_dataset_from_parquet_h3,
    load_sinr_dataset_from_parquet,
    load_sinr_dataset_from_parquet_h3,
)


@tf.function
def apply_gradient(optimizer, model, x, ys, fake_x, pos_weight):
    with tf.GradientTape() as tape:
        # make predictions for the true training data
        yhat = model(x, training=True)
        # make predictions for the fake/bg training data
        fake_yhat = model(fake_x, training=True)

        loss = sinr_loss(ys, yhat, fake_yhat, pos_weight, 1.0)

    # do the neural network bits, calculate the gradients and update
    # the model weights
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss


@click.command()
@click.option("--config_file", required=True)
def train_model(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(project="geomodel_tf_sinr", config=config)

    if config["sampling"]["discretized_sampling"]:
        if config["dataset_type"] == "sinr":
            (locs, class_ids, unique_taxa, h3_idx) = load_sinr_dataset_from_parquet_h3(
                config["sinr_dataset"]["train_data"],
                h3_resolution=config["sampling"]["h3_resolution"]
            )
        else:
            (locs, class_ids, unique_taxa, h3_idx) = load_inat_dataset_from_parquet_h3(
                config["inat_dataset"]["spatial_data"],
                h3_resolution=config["sampling"]["h3_resolution"]
            )
    else:
        if config["dataset_type"] == "sinr":
            (locs, class_ids, unique_taxa) = load_sinr_dataset_from_parquet(
                config["sinr_dataset"]["train_data"]
            )
        elif config["dataset_type"] == "inat":
            (locs, class_ids, unique_taxa) = load_inat_dataset_from_parquet(
                config["inat_dataset"]["spatial_data"]
            )

    if config["inputs"]["covariates"] == "env":
        raster = np.load(config["bioclim_data"]).astype(np.float32)
    elif config["inputs"]["covariates"] == "elev":
        raster = np.load(config["elev_data"]).astype(np.float32)
    else:
        raster = None

    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)

    encoded_locs = encoder.encode(locs)
    num_classes = len(unique_taxa)

    fcnet = make_geo_model_net(
        num_classes=num_classes, num_input_feats=encoder.num_input_feats()
    )

    losses = []

    if config["sampling"]["discretized_sampling"]:
        ds, num_train_steps_per_epoch = make_subsampled_dataset(
            config["hard_cap"],
            encoded_locs,
            class_ids,
            config["batch_size"],
            config["shuffle_buffer_size"],
            h3_idx=h3_idx
        )
    else:
        ds, num_train_steps_per_epoch = make_subsampled_dataset(
            config["hard_cap"],
            encoded_locs,
            class_ids,
            config["batch_size"],
            config["shuffle_buffer_size"],
            h3_idx=None
        )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["initial_lr"],
        decay_rate=config["lr_decay"],
        decay_steps=num_train_steps_per_epoch,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    pos_weight = tf.constant(config["sinr_hyperparams"]["pos_weight"], dtype=tf.float32)

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}")
        epoch_losses = []

        print(f" optimizer lr is {optimizer.learning_rate.numpy()}")

        if config["subsample_each_epoch"] and epoch != 0:
            print(" re subsampling dataset")
            ds, _ = make_subsampled_dataset(
                config["hard_cap"],
                encoded_locs,
                class_ids,
                config["batch_size"],
                config["shuffle_buffer_size"],
            )

        pbar = tqdm.tqdm(total=num_train_steps_per_epoch, dynamic_ncols=True)
        for step, (x_batch_train, y_batch_train) in enumerate(ds):
            pbar.update()
            # make random samples, comes out as list of x,y pairs where x and y are in -1, 1
            rand_loc = make_rand_samples_tf(
                # make same size fake data as real data
                len(x_batch_train)
            )
            rand_loc = encoder.encode(rand_loc, normalize=False)

            loss = apply_gradient(
                optimizer,
                fcnet,
                x_batch_train,
                y_batch_train,
                rand_loc,
                pos_weight
            )

            global_step = (epoch * num_train_steps_per_epoch) + step
            if step % 10 == 0:
                log_entry = {
                    "batch_loss": loss,
                    "learning_rate": optimizer.learning_rate.numpy(),
                }
                wandb.log(log_entry, step=global_step, commit=True)

            pbar.set_description(f" Loss {loss:.4f}")

            epoch_losses.append(loss.numpy())

        pbar.close()
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        print(f" Epoch mean loss: {epoch_loss:.4f}")

    fcnet.save(config["model_save_name"])
    wandb.finish()


if __name__ == "__main__":
    train_model()
