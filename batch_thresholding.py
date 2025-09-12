import os
from pathlib import Path
import sys

import h3
import h3pandas
import numpy as np
import pandas as pd
import pydeck as pdk
from shapely.geometry import shape
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tqdm.auto import tqdm

from lib.models.tf_gp_model import TFGeoPriorModel
from lib.models.encoders import CoordEncoder

GEOMODEL_PATH = "/home/alex/export_eval_models/model_exports/tf_geoprior_2_20_r6_elevation.h5"
ELEVATION_RASTER_PATH = "/home/alex/sinr/data/env/elev_scaled.npy"
INPUT_ENCODING = "sinusoidal"
TAX_2_20_FILE = "/data-ssd/alex/datasets/vision-export-20250119070012-aka-2.20/taxonomy.csv"
SPATIAL_DATA_2_20_FILE = "/data-ssd/alex/datasets/vision-export-20250119070012-aka-2.20/spatial_data.parquet"

def main():
    ## 1. Load Model and Setup
    
    print("Loading model, encoder, taxonomy")
    
    model = TFGeoPriorModel(GEOMODEL_PATH)
    raster = np.load(ELEVATION_RASTER_PATH).astype(np.float32)
    encoder = CoordEncoder(INPUT_ENCODING, raster=raster)
    tax_2_20 = pd.read_csv(TAX_2_20_FILE)
    
    ## 2. Load the spatial training data
    
    print("loading spatial training data")
    
    spatial_train = pd.read_parquet(SPATIAL_DATA_2_20_FILE)
    
    spatial_train_h3 = spatial_train.h3.geo_to_h3(
        resolution=4,
        lat_col="latitude",
        lng_col="longitude"
    )
    spatial_train_h3.reset_index(inplace=True)

    ## 3. Make geo features for all h3 cells at resolution 4

    print("make geo feats for all h3 cells at r4")

    res0_cells = h3.get_res0_indexes()
    all_cells = set()
    for res0 in tqdm(res0_cells):
        children = h3.h3_to_children(res0, 4)
        all_cells.update(children)
        
    cells_df = pd.DataFrame({"h3_04": list(all_cells)})
    cells_df.set_index("h3_04", inplace=True)

    dfh3 = cells_df.h3.h3_to_geo()
    dfh3["lng"] = dfh3.geometry.x
    dfh3["lat"] = dfh3.geometry.y
    _ = dfh3.pop("geometry")
    dfh3.reset_index(inplace=True)

    locs = dfh3[["lng", "lat"]].values
    encoded_locs = encoder.encode(locs)
    geo_model_features = model.get_loc_emb(encoded_locs)

    ## 4. Get LPT-R thresholds for all taxa in the model

    print("get lpt-r thresholds for all taxa in the model")
    
    leaf_tax_2_20 = tax_2_20[~tax_2_20.leaf_class_id.isna()]
    all_spatial_grid_counts = spatial_train_h3.h3_04.value_counts()

    # we're going to index on leaf class id so drop nas and convert to int
    spatial_train_h3.dropna(subset=["leaf_class_id"], inplace=True)
    spatial_train_h3.leaf_class_id=spatial_train_h3.leaf_class_id.astype(int)
    
    # index on the leaf_class_id since we'll be selecting on it in a loop
    spatial_train_h3.set_index("leaf_class_id", inplace=True)

    thresholds = []
    for i in tqdm(range(len(leaf_tax_2_20)), dynamic_ncols=True):
        leaf_taxon = leaf_tax_2_20.iloc[i]
        target_taxon_id = int(leaf_taxon.taxon_id)
        target_class_id = int(leaf_taxon.leaf_class_id)
    
        # get predictions for this class
        preds = model.eval_one_class_from_feats(
            geo_model_features,
            class_of_interest=target_class_id
        )
        dfh3["preds"] = preds[0]
    
        # get training data for this class
        spatial_train_h3_target_taxon = spatial_train_h3.loc[target_class_id]
    
        # compute lpt-r threshold at 5%
        presence_preds = dfh3[dfh3['h3_04'].isin(spatial_train_h3_target_taxon['h3_04'])]
        lpt_r_threshold = np.percentile(presence_preds.preds, 5)
        
        # make presence absence dataframe
        presence_absence = pd.DataFrame({
            "background": all_spatial_grid_counts,
        })
        presence_absence = presence_absence.fillna(0)
        target_spatial_grid_counts = spatial_train_h3_target_taxon.h3_04.value_counts()  
        presence_absence["forground"] = target_spatial_grid_counts
        presence_absence["predictions"] = dfh3.set_index("h3_04")["preds"]
        presence_absence.forground = presence_absence.forground.fillna(0)
        presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]
        lpt_r_thresh_paper_technique = np.percentile(presences, 5)
    
        thresholds.append({
            "taxon_id": target_taxon_id,
            "leaf_class_id": target_class_id,
            "lpt-r at 5% (fast)": lpt_r_threshold,
            "lpt-r at 5% (paper)": lpt_r_thresh_paper_technique,
        })

    ## 5. validate that fast & paper implementations produce the same results
    thresholds_df = pd.DataFrame(thresholds)
    thresholds_df["same"] = (
        thresholds_df["lpt-r at 5% (fast)"] == thresholds_df["lpt-r at 5% (paper)"]
    )
    print(f"LPT-R implemenations produce same results: { thresholds_df.same.all()}")

    ## 6. save generated thresholds
    thresholds_df.to_csv("thresholds_2_20_grid_elevation_lpt_r.csv")
    

if __name__ == "__main__":
    main()



