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

from lib.geo_model_net import make_geo_model_net
from utils import make_rand_samples, get_idx_subsample_observations
from encoders import CoordEncoder


@tf.function
def neg_log(x):
    return -tf.math.log(x + 1e-5)

def apply_gradient(optimizer, model, x, ys, fake_x, pos_weight):
    with tf.GradientTape() as tape:
        # make predictions for the true training data
        yhat = model(x)
        # make predictions for the fake/bg training data
        fake_yhat = model(fake_x)

        # start getting the loss for the true training data
        # neg_log for 1-prediction is large (2.3ish) if the
        # model predicts it's there, low (approaching zero)
        # if the model predicts it's not there.
        # for each lat/lng and taxon pair, we want the model
        # to predict false for everything other than the target
        # taxon. this part constructs the loss for the everything
        # else
        loss_pos = neg_log(1.0 - yhat)
        # print("before weighting, loss_poss mean is {}".format(tf.reduce_mean(loss_pos)))

        # now find the indices for the target taxa in the predictions
        inds = tf.constant(range(len(ys)), dtype="int64")
        inds = tf.stack([inds, ys], axis=1)
        # get the predictions for the target taxa
        pos_preds = tf.gather_nd(yhat, inds)
        # construct the loss for this term
        # this will be high (2300) if the model predicts
        # that it's not there, and low (100s) if the model
        # predicts it is there. because this dwards the other parts
        # of the loss, it's what the model will focus on minimizing
        # if the organism is there, predict that it's there
        newvals = pos_weight * neg_log(pos_preds)

        # this is tf's way of updating a tensor, not quite in place
        # since tf will make a new tensor for us by merginig the
        # initial values of loss_pos with the newvalues at the indices
        # provided
        loss_pos = tf.tensor_scatter_nd_update(loss_pos, [inds], [newvals])
        # print("after weighting, loss_poss mean is {}".format(tf.reduce_mean(loss_pos)))

        # this is the loss for the background data. basically, no matter
        # what, we want fake_yhat to be as low as possible since it's
        # made up data
        loss_bg = neg_log(1.0 - fake_yhat)
        # print("loss_bg mean is {}".format(tf.reduce_mean(loss_bg)))

        # get the means of the true and bg/fake loss, then sum them
        loss_value = tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_bg)

    # do the neural network bits, calculate the gradients and update
    # the model weights
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss_value, tf.reduce_mean(loss_pos), tf.reduce_mean(loss_bg)


def load_inat_dataset_from_parquet(spatial_data_file):
    print("inat style dataset")
    print(" reading parquet")
    spatial_data = pd.read_parquet(spatial_data_file)
    spatial_data = spatial_data.dropna(subset="leaf_class_id")
    print(" extracting locs")
    locs = np.vstack((
        spatial_data["longitude"].values,
        spatial_data["latitude"].values
    )).T.astype(np.float32)

    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)
    unique_taxa, _ = np.unique(taxon_ids, return_inverse=True)
 
    print(" extracting spatial class ids")
    class_ids = spatial_data["spatial_class_id"].values.astype(int)
    print(f" found {len(unique_taxa)} unique taxa")

    return locs, class_ids, unique_taxa

def load_sinr_dataset_from_parquet(file):
    print("sinr style dataset")
    print(" reading parquet")
    spatial_data = pd.read_parquet(file)
    
    print(" extracting locs")
    locs = np.vstack((
        spatial_data["longitude"].values, 
        spatial_data["latitude"].values
    )).T.astype(np.float32)
    
    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)

    print(" making class_ids")
    unique_taxa, class_ids = np.unique(taxon_ids, return_inverse=True)
    print(f" found {len(unique_taxa)} unique taxa")

    return locs, class_ids, unique_taxa


def make_subsampled_dataset(hard_cap, encoded_locs, class_ids, batch_size):
    ss_idx = get_idx_subsample_observations(
        class_ids,
        hard_cap=hard_cap
    )
    locs_ss = np.array(encoded_locs)[ss_idx]
    class_ids_ss = np.array(class_ids)[ss_idx]
    num_train_steps_per_epoch = locs_ss.shape[0] // batch_size
    ds = tf.data.Dataset.from_tensor_slices(
        (locs_ss, class_ids_ss)
    )
    ds = ds.shuffle(buffer_size=ds.cardinality())
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, num_train_steps_per_epoch

@click.command()
@click.option("--config_file", required=True)
def train_model(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if config["dataset_type"] == "sinr":
        (locs, class_ids, unique_taxa) = load_sinr_dataset_from_parquet(
            config["sinr_dataset"]["train_data"]
        )
    elif config["dataset_type"] == "inat":
        (locs, class_ids, unique_taxa) = load_inat_dataset_from_parquet(
            config["inat_dataset"]["spatial_data"]
        )
    else:
        assert False, f"unsupported dataset type {config['dataset_type']}"

    encoder = CoordEncoder()
    encoded_locs = encoder.encode(locs)    
    num_classes = len(unique_taxa)

    fcnet = make_geo_model_net(
        num_classes=num_classes,
        num_input_feats=config["num_input_feats"]
    )

    losses = []
    pos_losses = []
    bg_losses = []

    ds, num_train_steps_per_epoch = make_subsampled_dataset(
        config["hard_cap"],
        encoded_locs, 
        class_ids, 
        config["batch_size"]
    )
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["initial_lr"],
        decay_rate=config["lr_decay"],
        decay_steps=num_train_steps_per_epoch,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(config["num_epochs"]):    
        print(f"Epoch {epoch+1}")
        epoch_losses = []
        epoch_pos_losses = []
        epoch_bg_losses = []
        
        print(f" optimizer lr is {optimizer.learning_rate.numpy()}")
        
        if config["subsample_each_epoch"] and epoch != 0:
            print(" re subsampling dataset")
            ds, _ = make_subsampled_dataset(
                config["hard_cap"],
                encoded_locs,
                class_ids,
                config["batch_size"]
            )

        pbar = tqdm.tqdm(total=num_train_steps_per_epoch, dynamic_ncols=True)      
        for step, (x_batch_train, y_batch_train) in enumerate(ds):
            pbar.update()
            # make random samples, comes out as list of x,y pairs where x and y are in -1, 1
            rand_loc = make_rand_samples(
                # make same size fake data as real data
                len(x_batch_train)
            )
            rand_loc = encoder.encode(rand_loc, normalize=False)
    
            (loss, loss_pos, loss_bg) = apply_gradient(
                optimizer, 
                fcnet, 
                x_batch_train, 
                y_batch_train, 
                rand_loc,
                config["sinr_hyperparams"]["pos_weight"]
            )
   
                 
            pbar.set_description(f" Loss {loss:.4f}")

            epoch_losses.append(loss.numpy())
            epoch_pos_losses.append(loss_pos.numpy())
            epoch_bg_losses.append(loss_bg.numpy())
        
        pbar.close() 
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        pos_losses.append(np.mean(epoch_pos_losses))
        bg_losses.append(np.mean(epoch_bg_losses))
        print(f" Epoch mean loss: {epoch_loss:.4f}")

    fcnet.save(config["model_save_name"])


if __name__ == "__main__":
    train_model()
