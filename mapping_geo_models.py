import colorsys
import math
import os
from pathlib import Path

import click
import h3pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from lib.models.encoders import CoordEncoder
from lib.models.tf_gp_model import TFGeoPriorModel
from sinr_paper_evals import load_class_to_taxa_inat_dataset, load_class_to_taxa_sinr_dataset

def coord_grid(grid_size):
    feats = np.zeros(
        (grid_size[0], grid_size[1], 2),
        dtype=np.float32
    )
    mg = np.meshgrid(
        np.linspace(-180, 180, feats.shape[1]),
        np.linspace(90, -90, feats.shape[0]),
    )
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]

    return feats.reshape(feats.shape[0] * feats.shape[1], 2)
    

@click.command()
@click.option("--config_file", type=str, required=True)
@click.option("--taxon_ids", type=str, required=True)
@click.option("--output_dir", type=str, required=True)
def make_map(config_file, taxon_ids, output_dir):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # no gpu for eval
    tf.config.set_visible_devices([], "GPU")
    for device in tf.config.get_visible_devices():
        assert device.device_type != "GPU"

    if config["inputs"]["covariates"] == "env":
        raster = np.load(config["bioclim_data"]).astype(np.float32)
    elif config["inputs"]["covariates"] == "elev":
        raster = np.load(config["elev_data"]).astype(np.float32)
    else:
        raster = None

    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)
    model = TFGeoPriorModel(config["model_save_name"])

    print("loading class to taxa")
    if config["dataset_type"] == "sinr":
        class_to_taxa = load_class_to_taxa_sinr_dataset(
            config["sinr_dataset"]["train_data"]
        )
    elif config["dataset_type"] == "inat":
        class_to_taxa = load_class_to_taxa_inat_dataset(
            config["inat_dataset"]["taxonomy_data"]
        )

    ocean_mask_file = config["mapping"]["mask"]
    mask = np.load(ocean_mask_file)
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    locs = coord_grid(mask.shape)
    if config["mapping"]["mask_ocean"]:
        locs = locs[mask_inds, :]
    
    encoded_locs = encoder.encode(locs)

    print("getting geo model features")    
    geo_model_features = model.get_loc_emb(encoded_locs)

    if "," in taxon_ids:
        taxon_ids = [int(x) for x in taxon_ids.split(",")]
    else:
        taxon_ids = [int(taxon_ids)]

    os.makedirs(output_dir, exist_ok=True)

    for taxon_id in taxon_ids:
        print(f"mapping {taxon_id}")
        class_id = np.squeeze(np.where(class_to_taxa == taxon_id))
        print(f"looking for class id {class_id}")
        preds = model.eval_one_class_from_feats(
            geo_model_features, 
            class_id
        )
        op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan
        if config["mapping"]["mask_ocean"]:
            op_im[mask_inds] = preds
        else:
            op_im = preds

        op_im = op_im.reshape((mask.shape[0], mask.shape[1]))
        op_im = np.ma.masked_invalid(op_im)

        cmap = plt.cm.plasma
        cmap.set_bad(color="none")
        vmax = np.max(op_im)
        
        filename = f"{output_dir}/{taxon_id}_testing.png"
        print(f"saving to {filename}")
        plt.imsave(filename, arr=op_im, vmin=0, vmax=vmax, cmap=cmap)
    
   
 

if __name__ == "__main__":
    make_map()
