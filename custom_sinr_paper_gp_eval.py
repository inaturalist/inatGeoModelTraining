import hashlib
import math
import os
from pathlib import Path

import h3pandas
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

from lib.models.geo_model_net import ResLayer


def encode_location_elevation(lat, lng, elev):
    norm_lat = np.array([float(lat)]) / 90.0
    norm_lng = np.array([float(lng)]) / 180.0
    norm_loc = tf.stack([norm_lng, norm_lat], axis=1)
    norm_loc = tf.cast(norm_loc, tf.float32)
    elevation = tf.expand_dims(float(elev), axis=0)
    elevation = tf.expand_dims(elevation, axis=0)
    return tf.concat([
        tf.sin(norm_loc * math.pi),
        tf.cos(norm_loc * math.pi),
        elevation,
    ], axis=1)

tax_2_17 = pd.read_csv(
    "/data-ssd/alex/datasets/vision-export-20240929050006-aka-2.17/taxonomy.csv"
)
gp_model_2_17 = tf.keras.models.load_model(
    #"/home/alex/export_eval_models/model_exports/tf_geoprior_2_17_r6_elevation.h5",
    "/home/alex/inatGeoModelTraining/saved_model_2_17_ds_sinr_approach_coords_elev.h5",
    custom_objects={"ResLayer": ResLayer},
    compile=False
)
elevations = pd.read_csv(
    "/home/alex/elevation_r4_5m.csv",
    index_col="h3_04"
)
above_sea_level_norm = elevations.elevation[elevations.elevation>0] / np.max(elevations.elevation)
below_sea_level_norm = (elevations.elevation[elevations.elevation<0] / np.min(elevations.elevation)) * -1
norm_elev =pd.concat([
    below_sea_level_norm,
    above_sea_level_norm
])
norm_elev_df = pd.DataFrame({
    "elevation": norm_elev
})

gp_eval_dir = Path("/home/alex/sinr/data/eval/geo_prior")
sinr_gp_model_meta = pd.read_csv(
    gp_eval_dir / "geo_prior_model_meta.csv"
)
gp_model_preds = np.load(
    gp_eval_dir / "geo_prior_model_preds.npz"
)

tax_in_gp_task = set(sinr_gp_model_meta.taxon_id.unique())

leaf_tax_2_17 = tax_2_17[~tax_2_17.leaf_class_id.isna()]
tax_in_gp_model = set(leaf_tax_2_17.taxon_id.unique())

tax_in_both_models = tax_in_gp_task.intersection(tax_in_gp_model)

sinr_gp_model_meta_dfh3 = sinr_gp_model_meta.h3.geo_to_h3(
    resolution=4,
    lat_col="latitude",
    lng_col="longitude",
)
sinr_gp_model_meta_elev = pd.merge(
    sinr_gp_model_meta_dfh3,
    norm_elev_df,
    how="inner",
    left_index=True,
    right_index=True
)

all_probs = gp_model_preds["probs"]
all_inds = gp_model_preds["inds"]
all_taxon_ids = gp_model_preds["model_to_taxa"]

leaf_tax_2_17 = tax_2_17[~tax_2_17.leaf_class_id.isna()]
leaf_tax_2_17 = leaf_tax_2_17.sort_values(
    "spatial_class_id"
)
leaf_tax_2_17.set_index(
    "spatial_class_id",
    inplace=True
)


seen = 0
cv_correct = 0
comb_correct = 0
#SAMPLE_SIZE = len(sinr_gp_model_meta_elev)
SAMPLE_SIZE = 10

indices = list(range(len(sinr_gp_model_meta_elev)))[:100_000]
rng = np.random.default_rng(42)
np.random.shuffle(indices)


for index in tqdm(range(SAMPLE_SIZE), dynamic_ncols=True):
    i = indices[index]

    row = sinr_gp_model_meta_elev.iloc[i]
    if not row.taxon_id in tax_in_both_models:
        continue
    print(row.taxon_id)

    probs = all_probs[i]
    inds = all_inds[i]

    taxon_ids = all_taxon_ids[inds]
    probs_df = pd.DataFrame({
        "probs": probs,
        "inds": inds,
        "taxon_id": taxon_ids,
    })
    probs_yhat = np.argmax(probs)
    idx_yhat = inds[probs_yhat]
    tid_yhat = all_taxon_ids[probs_yhat]

    cv_top1_taxon_id = int(
        probs_df.sort_values(
            "probs", ascending=False
        ).iloc[0].taxon_id
    )
    print(probs_yhat, idx_yhat, tid_yhat, cv_top1_taxon_id)

    encoded_loc = encode_location_elevation(
        row["latitude"],
        row["longitude"],
        row["elevation"],
    )
    gp_preds = gp_model_2_17.predict(encoded_loc, verbose=0)
    gp_preds_df = pd.DataFrame({
        "gp_probs": gp_preds[0],
        "spatial_class_id": leaf_tax_2_17.index,
        "taxon_id": leaf_tax_2_17.taxon_id,
    })

    comb_df = pd.merge(
        probs_df,
        gp_preds_df,
        how="inner",
        left_on="taxon_id",
        right_on="taxon_id",
    )

    comb_df["comb_score"] = comb_df["probs"] * comb_df["gp_probs"]
    comb_top1_taxon_id = int(
        comb_df.sort_values(
            "comb_score", ascending=False
        ).iloc[0].taxon_id
    )

    seen += 1
    if cv_top1_taxon_id == row.taxon_id:
        cv_correct += 1
    if comb_top1_taxon_id == row.taxon_id:
        comb_correct += 1

print(cv_correct / seen)
print(comb_correct / seen)

print(f"improvement: {(comb_correct/seen - cv_correct/seen) * 100:.2f}")

