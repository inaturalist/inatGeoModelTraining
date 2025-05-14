"""
make_thresholds.py

Generates per-taxon threshold values for a trained geospatial model using one of several thresholding strategies.

Usage:
    python make_thresholds.py \
        --train_config_file configs/config_train_inat_2_21.yaml \
        --thresholding_strategy lpt_r \
        --lpt_r_percentile 5 \
        --output_path results/thresholds_inat_2_21_lpt_r.csv

Supported strategies:
    - lpt_r (requires --lpt_r_percentile)
    - (TODO: lpt, scott_pa, fixed)

Outputs:
    CSV with columns: taxon_id, threshold_type, threshold_value
"""
import click
import h3
import h3pandas
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import yaml

from lib.models.tf_gp_model import TFGeoPriorModel
from lib.models.encoders import CoordEncoder

@click.command()
@click.option("--train_config_file", type=click.Path(exists=True), required=True, help="Path to training config YAML file.")
@click.option("--thresholding_strategy", type=str, required=True, help="Thresholding method to use (lpt_r, lpt, scott_pa, fixed).")
@click.option("--fixed_threshold", type=float, required=False, help="Threshold value if using 'fixed' strategy.")
@click.option("--lpt_r_percentile", type=int, required=False, help="Percentile value for LPT-R (e.g., 5 for 5th percentile).")
@click.option("--output_path", type=click.Path(), required=True, help="Path to output CSV file.")
@click.option("--stop_after", type=int, required=False, help="Optional: stop after evaluating N taxa (for debugging).")
def make_thresholds(train_config_file, thresholding_strategy, fixed_threshold, lpt_r_percentile, output_path, stop_after):
    valid_thresholding_strats = ["lpt_r", "lpt", "scott_pa", "fixed"]

    assert thresholding_strategy in valid_thresholding_strats, f"unsupported thresholding strategy {thresholding_strategy}, supported techniques are {valid_thresholding_strats}"

    if thresholding_strategy == "lpt_r":
        assert lpt_r_percentile is not None, "if using lpt-r, a percentile is required"
    
    with open(train_config_file, "r") as f:
        train_config = yaml.safe_load(f)

    # load model & basics
    print("loading model, encoder, taxonomy...")
    model = TFGeoPriorModel(train_config.get("model_save_name"))

    if train_config.get("inputs").get("covariates") == "env":
        assert train_config.get("bioclim_data") is not None, "if using env, need bioclim data"
        raster = np.load(train_config.get("bioclim_data")).astype(np.float32)
    elif train_config.get("inputs").get("covariates") == "elev":
        assert train_config.get("elev_data") is not None, "if using elev, need elevation data"
        raster = np.load(train_config.get("elev_data")).astype(np.float32)
    else:
        raster = None
    encoder = CoordEncoder(train_config.get("inputs").get("loc_feat_encoding"), raster)

    if train_config.get("dataset_type") == "inat":
        train_dataset = train_config.get("inat_dataset")
    elif train_config.get("dataset_type") == "sinr":
        train_dataset = train_config.get("sinr_dataset")
    else:
        raise NotImplementedError(f"{train_config.get('dataset_type')} dataset type not implemented yet.")

    tax = pd.read_csv(train_dataset.get("taxonomy_data"))

    # load training data
    if thresholding_strategy in ["lpt", "lpt_r", "scott_pa"]:
        print("loading spatial training data...")
        spatial_train = pd.read_parquet(train_dataset.get("train_data"))
        if train_config.get("dataset_type") == "sinr":
            # we need to manually add the leaf_class_id column in spatial_train
            print("adding leaf_class_id to sinr style spatial data dataset")
            spatial_train = pd.merge(
                spatial_train,
                tax[["taxon_id", "leaf_class_id"]],
                left_on="taxon_id",
                right_on="taxon_id",
                how="left",
            )
       
        print("discretizing spatial training data at resolution 4...") 
        spatial_train_h3 = spatial_train.h3.geo_to_h3(
            resolution=4,
            lat_col="latitude",
            lng_col="longitude",
        )
        spatial_train_h3.reset_index(inplace=True)
        spatial_train_h3.set_index("leaf_class_id", inplace=True)

    # generate geo features for all cells at resolution 4 for evaluation
    print("make geo features for all h3 cells at r4")
    res0_cells = h3.get_res0_indexes()
    all_cells = set()
    for res0 in tqdm(res0_cells, dynamic_ncols=True):
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
    geo_model_feats = model.get_loc_emb(encoded_locs)
            
    # make thresholds for all taxa in the model
    print("making thresholds...")
    leaf_tax = tax[~tax.leaf_class_id.isna()]
    thresholds = []
    for i in tqdm(range(len(leaf_tax)), dynamic_ncols=True):
        leaf_taxon = leaf_tax.iloc[i]
        target_taxon_id = int(leaf_taxon.taxon_id)
        target_class_id = int(leaf_taxon.leaf_class_id)

        preds = model.eval_one_class_from_feats(
            geo_model_feats,
            class_of_interest=target_class_id
        )
        
        if thresholding_strategy == "lpt_r": 
            assert lpt_r_percentile is not None, "lpt_r_percentile must be provided for 'lpt_r' strategy"

            dfh3["preds"] = preds[0]
            spatial_train_h3_target_taxon = spatial_train_h3.loc[target_class_id]
            presence_preds = dfh3[dfh3["h3_04"].isin(spatial_train_h3_target_taxon["h3_04"])]
            lpt_r_threshold = np.percentile(presence_preds.preds, lpt_r_percentile)
            thresholds.append({
                "taxon_id": target_taxon_id,
                "threshold_type": f"lpt_r_at_{lpt_r_percentile}_pct",
                "threshold_value": lpt_r_threshold,
            })
        elif thresholding_strategy == "fixed":
            assert fixed_threshold is not None, "fixed_threshold must be provided for 'fixed' strategy"
            thresholds.append({
                "taxon_id": target_taxon_id,
                "threshold_type": f"fixed_{fixed_threshold}",
                "threshold_value": fixed_threshold,
            })
        elif thresholding_strategy in ["lpt", "scott_pa"]:
            raise NotImplementedError(f"{thresholding_strategy} strategy not implemented yet.")    

        if stop_after is not None and i >= stop_after:
            break
   
    print("saving thresholds...")
    thresholds_df = pd.DataFrame(thresholds)
    thresholds_df.to_csv(
        output_path, index=False
    )
        



if __name__ == "__main__":
    make_thresholds()
