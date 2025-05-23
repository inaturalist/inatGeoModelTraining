import click
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tqdm.auto import tqdm
import yaml

from lib.models.encoders import CoordEncoder
from lib.models.geo_model_net import ResLayer
from lib.models.tf_gp_model import TFGeoPriorModel
from lib.eval.snt_eval import EvaluatorSNT
from lib.eval.iucn_eval import EvaluatorIUCN
from lib.eval.geo_prior_eval import EvaluatorGeoPrior
from lib.eval.geo_feat_eval import EvaluatorGeoFeatures


def load_class_to_taxa_inat_dataset(taxonomy):
    tax = pd.read_csv(taxonomy)
    leaf_tax = tax[~tax.leaf_class_id.isna()]
    leaf_tax = leaf_tax.sort_values("spatial_class_id")
    class_to_taxa = leaf_tax["taxon_id"].values.astype(int)
    return class_to_taxa


def load_class_to_taxa_sinr_dataset(file):
    print(" reading parquet")
    spatial_data = pd.read_parquet(file)

    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)

    class_to_taxa, _ = np.unique(taxon_ids, return_inverse=True)
    return class_to_taxa


@click.command()
@click.option("--config_file", required=True)
def run_eval(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # no gpu for eval - iucn in particular won't fit on a gpu
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"

    if config["inputs"]["covariates"] == "env":
        raster = np.load(config["bioclim_data"]).astype(np.float32)
    elif config["inputs"]["covariates"] == "elev":
        raster = np.load(config["elev_data"]).astype(np.float32)
    else:
        raster = None
    enc = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)
    model = TFGeoPriorModel(config["model_save_name"])

    evals_to_do = config["sinr_eval"]["evals_to_do"]

    print("loading class to taxa")
    if config["dataset_type"] == "sinr":
        class_to_taxa = load_class_to_taxa_sinr_dataset(
            config["sinr_dataset"]["train_data"]
        )
    elif config["dataset_type"] == "inat":
        class_to_taxa = load_class_to_taxa_inat_dataset(
            config["inat_dataset"]["taxonomy_data"]
        )

    if "snt" in evals_to_do:
        print("snt")
        evaluator = EvaluatorSNT(
            npy_path=config["sinr_eval"]["snt_npy_path"],
            split=config["sinr_eval"]["snt_split"],
            val_frac=config["sinr_eval"]["snt_val_frac"],
            class_to_taxa=class_to_taxa,
        )
        results = evaluator.run_eval(model, enc)
        evaluator.report(results)
        print()

    if "iucn" in evals_to_do:
        print("iucn")
        evaluator = EvaluatorIUCN(
            json_path=config["sinr_eval"]["iucn_json_path"], class_to_taxa=class_to_taxa
        )
        results = evaluator.run_eval(model, enc)
        evaluator.report(results)
        print()

    if "geo_prior" in evals_to_do:
        print("geo prior")
        evaluator = EvaluatorGeoPrior(
            preds_path=config["sinr_eval"]["gp_preds_path"],
            meta_path=config["sinr_eval"]["gp_meta_path"],
            batch_size=config["sinr_eval"]["gp_batch_size"],
            class_to_taxa=class_to_taxa,
        )
        results = evaluator.run_eval(model, enc)
        evaluator.report(results)
        print()

    if "geo_feat" in evals_to_do:
        print("geo feat")
        evaluator = EvaluatorGeoFeatures(
            data_path=config["sinr_eval"]["gf_data_dir"],
            mask=config["sinr_eval"]["gf_mask"],
            class_to_taxa=class_to_taxa,
        )
        results = evaluator.run_eval(model, enc)
        evaluator.report(results)
        print()


if __name__ == "__main__":
    run_eval()
