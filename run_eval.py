"""
run_eval.py

Run evaluation.

Usage:
    python run_eval.py \
        --train_config_file configs/config_train_inat_2_21.yaml \
        --thresholds_file thresholds/thresh_inat_2_21_lpt_5.csv \
        --thresh_value_colname threshold_value \
        --eval_dataset iucn \
        --iucn_file ~/sinr/data/eval/iucn/iucn_res_5.json \
        --output_path results/eval_inat_2_21_lpt_5_iucn.csv
    
Supported evals datasets:
    - iucn, snt, inat

Outputs:
    CSV with columns: taxon_id, threshold, precision, recall, f1
"""
import json
from pathlib import Path
import os

import click
import h3
import h3pandas
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm
import yaml

from lib.models.tf_gp_model import TFGeoPriorModel
from lib.models.encoders import CoordEncoder

def prepare_encoder_and_model(train_config):
    model = TFGeoPriorModel(train_config.get("model_save_name"))
    cov_type = train_config["inputs"]["covariates"]
    if cov_type == "env":
        raster = np.load(train_config["bioclim_data"]).astype(np.float32)
    elif cov_type == "elev":
        raster = np.load(train_config["elev_data"]).astype(np.float32)
    else:
        raster = None
    encoder = CoordEncoder(train_config["inputs"]["loc_feat_encoding"], raster)
    return model, encoder

def get_leaf_taxa(train_config):
    dataset_type = train_config["dataset_type"]
    if dataset_type == "inat":
        dataset = train_config["inat_dataset"]
    elif dataset_type == "sinr":
        dataset = train_config["sinr_dataset"]
    else:
        raise NotImplementedError(f"{dataset_type} dataset not supported.")
    tax = pd.read_csv(dataset["taxonomy_data"])
    return tax[~tax.leaf_class_id.isna()]

def load_iucn_ranges(iucn_range_path):
    with open(iucn_range_path, "r") as f:
        data = json.load(f)
    obs_locs = np.array(data["locs"], dtype=np.float32)
    taxa = [int(tt) for tt in data["taxa_presence"].keys()]
    taxa_presence = data["taxa_presence"]
    return obs_locs, taxa, taxa_presence

def load_snt_ranges(snt_range_path):
    data = np.load(snt_range_path, allow_pickle=True)
    data = data.item()

    species_ids = data["taxa"]
    loc_indices_per_species = data["loc_indices_per_species"]
    labels_per_species = data["labels_per_species"]
    obs_locs = data["obs_locs"]

    return obs_locs, species_ids, loc_indices_per_species, labels_per_species 

def score_taxon(preds, taxon_threshold, y_true):
    y_pred = (preds >= taxon_threshold).astype(int)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return p, r, f1


def evaluate_snt(model, encoder, leaf_tax, thresholds, snt_file, thresh_value_colname, max_taxa):
    obs_locs, taxa, loc_indices_per_species, labels_per_species = load_snt_ranges(snt_file)

    encoded_locs = encoder.encode(obs_locs)
    geo_model_feats = model.get_loc_emb(encoded_locs)
    results = []

    for taxon_id in tqdm(taxa[:max_taxa] if max_taxa else taxa, dynamic_ncols=True):
        if taxon_id not in leaf_tax.taxon_id.values:
            print(f"taxon id {taxon_id} not in model.")
            continue
        if taxon_id not in thresholds.taxon_id.values:
            print(f"taxon id {taxon_id} not in thresholds.")
            continue

        target_class_id = int(leaf_tax[leaf_tax.taxon_id==taxon_id].iloc[0].leaf_class_id)
        taxon_threshold = thresholds[thresholds.taxon_id == taxon_id].iloc[0][thresh_value_colname]
        preds = model.eval_one_class_from_feats(
            geo_model_feats, class_of_interest=target_class_id
        )[0]

        taxon_index = np.where(np.array(taxa) == taxon_id)[0][0]
        cur_loc_indices = np.array(loc_indices_per_species[taxon_index])
        pred = preds[cur_loc_indices]
        y_test = np.array(labels_per_species[taxon_index])

        p, r, f1 = score_taxon(pred, threshold, y_test)
        results.append({
            "taxon_id": taxon_id,
            "threshold": taxon_threshold,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    return pd.DataFrame(results)

def evaluate_iucn(model, encoder, leaf_tax, thresholds, iucn_file, thresh_value_colname, max_taxa):

    obs_locs, taxa, tp = load_iucn_ranges(iucn_file)

    encoded_locs = encoder.encode(obs_locs)
    geo_model_feats = model.get_loc_emb(encoded_locs)
   
    results = []

    for taxon_id in tqdm(taxa[:max_taxa] if max_taxa else taxa, dynamic_ncols=True): 
        if taxon_id not in leaf_tax.taxon_id.values:
            continue

        if taxon_id not in thresholds.taxon_id.values:
            continue

        target_class_id = int(leaf_tax[leaf_tax.taxon_id==taxon_id].iloc[0].leaf_class_id)
        taxon_threshold = thresholds[thresholds.taxon_id == taxon_id].iloc[0][thresh_value_colname]
        preds = model.eval_one_class_from_feats(
            geo_model_feats, class_of_interest=target_class_id
        )[0]

        true_presences = np.zeros(obs_locs.shape[0], dtype=np.float32)
        tp_taxon = tp[str(taxon_id)]
        true_presences[tp_taxon] = 1.0

        p, r, f1 = score_taxon(preds, taxon_threshold, true_presences)
        results.append({
            "taxon_id": taxon_id,
            "threshold": taxon_threshold,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    return pd.DataFrame(results)


def evaluate_inat(model, encoder, leaf_tax, thresholds, inat_taxon_range_recalls, inat_taxon_range_csvs, inat_recall_threshold, thresh_value_colname, max_taxa):
    trr = pd.read_csv(inat_taxon_range_recalls)
    taxa = trr.taxon_id.values
  
    # generate geo features for all cells at resolution 4 for evaluation
    print("make geo features for all h3 cells at r4")
    res0_cells = h3.get_res0_indexes()
    all_cells = set()
    for res0 in tqdm(res0_cells, dynamic_ncols=True):
        children = h3.h3_to_children(res0, 4)
        all_cells.update(children)
    cells_df = pd.DataFrame({"h3_04": sorted(all_cells)})
    cells_df.set_index("h3_04", inplace=True)
    dfh3 = cells_df.h3.h3_to_geo()
    dfh3["lng"] = dfh3.geometry.x
    dfh3["lat"] = dfh3.geometry.y
    _ = dfh3.pop("geometry")
    dfh3.reset_index(inplace=True)
    obs_locs = dfh3[["lng", "lat"]].values
    encoded_locs = encoder.encode(obs_locs)
    geo_model_feats = model.get_loc_emb(encoded_locs)
    results = []
   
    for taxon_id in tqdm(taxa[:max_taxa] if max_taxa else taxa, dynamic_ncols=True):
        if taxon_id not in leaf_tax.taxon_id.values:
            continue
        if taxon_id not in thresholds.taxon_id.values:
            continue
        tr_csv_dir = Path(inat_taxon_range_csvs) / f"{taxon_id}.csv"
        if not os.path.exists(tr_csv_dir):
            continue
        trr_recall = trr[trr.taxon_id == taxon_id].iloc[0].recall 
        if trr_recall < inat_recall_threshold:
            continue
      
        target_class_id = int(leaf_tax[leaf_tax.taxon_id==taxon_id].iloc[0].leaf_class_id)
        taxon_threshold = thresholds[thresholds.taxon_id == taxon_id].iloc[0][thresh_value_colname]
        preds = model.eval_one_class_from_feats(
            geo_model_feats, class_of_interest=target_class_id
        )[0]

        known_cells = pd.read_csv(tr_csv_dir, header=None)[0].values
        
        dfh3["preds"] = preds
        dfh3["binary_preds"] = (dfh3["preds"] >= taxon_threshold).astype(int)

        dfh3["y_true"] = dfh3["h3_04"].isin(known_cells).astype(int)
        dfh3["y_pred"] = dfh3["binary_preds"]

        y_true = dfh3["y_true"]
        y_pred = dfh3["y_pred"]

        p, r, f1 = score_taxon(preds, taxon_threshold, y_true)
        results.append({
            "taxon_id": taxon_id,
            "threshold": taxon_threshold,
            "precision": p,
            "recall": r,
            "f1": f1,
        })
    
    return pd.DataFrame(results)



@click.command()
@click.option("--train_config_file", type=click.Path(exists=True), required=True, help="Path to training config YAML file.")
@click.option("--thresholds_file", type=click.Path(exists=True), required=True, help="Path to computed thresholds file.")
@click.option("--thresh_value_colname", type=str, required=True, help="Name of the column containing the threshold.")
@click.option("--eval_dataset", type=str, required=True, help="Eval dataset (snt, iucn, inat")
@click.option("--iucn_file", type=click.Path(exists=True), required=False, help="Path to iUCN dataset json file.")
@click.option("--snt_file", type=click.Path(exists=True), required=False, help="Path to snt dataset numpy file.")
@click.option("--inat_taxon_range_recalls", type=click.Path(exists=True), required=False, help="Path to inat taxon range recalls.")
@click.option("--inat_taxon_range_csvs", type=click.Path(exists=True), required=False, help="Path to inat taxon range csvs.")
@click.option("--inat_recall_threshold", type=float, required=False, help="Recall threshold for inat taxa.")
@click.option("--max_taxa", type=int, required=False, help="Optional: limit number of taxa for debugging.")
@click.option("--output_path", type=click.Path(), required=True, help="Where to save per-taxon evaluation results.")
def run_eval(train_config_file, thresholds_file, thresh_value_colname, eval_dataset, iucn_file, snt_file, inat_taxon_range_recalls, inat_taxon_range_csvs, inat_recall_threshold, max_taxa, output_path):

    with open(train_config_file, "r") as f:
        train_config = yaml.safe_load(f)

    print("loading model, encoder, taxonomy...")
    model, encoder = prepare_encoder_and_model(train_config)
    leaf_tax = get_leaf_taxa(train_config)

    thresholds = pd.read_csv(thresholds_file)

    results = []

    if eval_dataset == "inat":
        assert inat_taxon_range_recalls is not None, "inat_taxon_range_recalls must be provided for inat eval"
        assert inat_taxon_range_csvs is not None, "inat_taxon_range_csvs must be provided for inat eval"
        assert inat_recall_threshold is not None, "inat_recall_threshold must be provided for inat eval"
        results_df = evaluate_inat(model, encoder, leaf_tax, thresholds, inat_taxon_range_recalls, inat_taxon_range_csvs, inat_recall_threshold, thresh_value_colname, max_taxa)

    elif eval_dataset == "snt":
        assert snt_file is not None, "snt_file must be provided for snt eval"
        results_df = evaluate_snt(model, encoder, leaf_tax, thresholds, snt_file, thresh_value_colname, max_taxa)
 
    elif eval_dataset == "iucn":
        assert iucn_file is not None, "iucn_file must be provided for iucn eval"
        results_df = evaluate_iucn(model, encoder, leaf_tax, thresholds, iucn_file, thresh_value_colname, max_taxa) 

    else:
        raise NotImplementedError(f"{eval_dataset} not implemented yet.")

    results_df.to_csv(output_path)
    print(results_df.describe())


if __name__ == "__main__":
    run_eval()
