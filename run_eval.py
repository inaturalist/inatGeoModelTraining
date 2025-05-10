"""
run_eval.py

Run evaluation.

Usage:
    python run_eval.py \
        --train_config_file configs/config_train_inat_2_21.yaml \
        --thresholds_file thresholds/thresh_inat_2_21_lpt_5.csv \
        --thresh_value_coluname threshold_value \
        --eval_dataset iucn \
        --iucn_file ~/sinr/data/eval/iucn/iucn_res_5.json \
        --output_path results/eval_inat_2_21_lpt_5_iucn.csv
    
Supported evals datasets:
    - iucn
    - (TODO: snt, iNat, iNat trimmed)

Outputs:
    CSV with columns: taxon_id, threshold, precision, recall, f1
"""
import json

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


def load_iucn_ranges(iucn_range_path):
    with open(iucn_range_path, "r") as f:
        data = json.load(f)
    obs_locs = np.array(data["locs"], dtype=np.float32)
    taxa = [int(tt) for tt in data["taxa_presence"].keys()]
    taxa_presence = data["taxa_presence"]
    return obs_locs, taxa, taxa_presence


@click.command()
@click.option("--train_config_file", type=click.Path(exists=True), required=True, help="Path to training config YAML file.")
@click.option("--thresholds_file", type=click.Path(exists=True), required=True, help="Path to computed thresholds file.")
@click.option("--thresh_value_colname", type=str, required=True, help="Name of the column containing the threshold.")
@click.option("--eval_dataset", type=str, required=True, help="Eval dataset (snt, iucn, iNat, iNat trimmed.")
@click.option("--iucn_file", type=click.Path(exists=True), required=False, help="Path to iUCN dataset json file.")
@click.option("--max_taxa", type=int, required=False, help="Optional: limit number of taxa for debugging.")
@click.option("--output_path", type=click.Path(), required=True, help="Where to save per-taxon evaluation results.")
def run_eval(train_config_file, thresholds_file, thresh_val_colname, eval_dataset, iucn_file, max_taxa, output_path):
    with open(train_config_file, "r") as f:
        train_config = yaml.safe_load(f)

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
        raise NotImplementedError(f"{train_config.get('dataset_type')} dataset not implemented yet.")
    tax = pd.read_csv(train_dataset.get("taxonomy_data"))

    leaf_tax = tax[~tax.leaf_class_id.isna()]

    thresholds = pd.read_csv(thresholds_file)

    if eval_dataset == "iucn":
        assert iucn_file is not None, "iucn_file must be provided for iucn eval"
        obs_locs, taxa, tp = load_iucn_ranges(iucn_file)

        encoded_locs = encoder.encode(obs_locs)
        geo_model_feats = model.get_loc_emb(encoded_locs)
       
        results = []

        for taxon_id in tqdm(taxa[:max_taxa] if max_taxa else taxa, dynamic_ncols=True): 
            if taxon_id not in leaf_tax.taxon_id.values:
                continue

            if taxon_id not in thresholds.taxon_id.values:
                continue

            threshold = thresholds[thresholds.taxon_id == taxon_id].iloc[0]
            taxon_threshold = threshold[thresh_val_colname]
            
            leaf_taxon = leaf_tax[leaf_tax.taxon_id==taxon_id].iloc[0]
            target_class_id = int(leaf_taxon.leaf_class_id)
            preds = model.eval_one_class_from_feats(
                geo_model_feats,
                class_of_interest=target_class_id
            )[0]
            
            true_presences = np.zeros(obs_locs.shape[0], dtype=np.float32)
            tp_taxon = tp[str(taxon_id)]
            true_presences[tp_taxon] = 1.0

            binarized_preds = (preds >= taxon_threshold)

            p = precision_score(true_presences, binarized_preds)
            r = recall_score(true_presences, binarized_preds)
            f1 = f1_score(true_presences, binarized_preds)
            results.append({
                "taxon_id": taxon_id,
                "threshold": taxon_threshold,
                "precision": p,
                "recall": r,
                "f1": f1,
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path)
        print(results_df)
        print(results_df.describe())
    else:
        raise NotImplementedError(f"{eval_dataset} not implemented yet.")



if __name__ == "__main__":
    run_eval()
