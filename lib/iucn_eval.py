import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tqdm.auto import tqdm

class EvaluatorIUCN:
    def __init__(self, json_path, class_to_taxa):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.obs_locs = np.array(
            self.data["locs"], dtype=np.float32
        )
        self.taxa = [
            int(tt) for tt in self.data["taxa_presence"].keys()
        ]
        self.class_to_taxa = class_to_taxa
       
    
    def run_eval(self, model, enc):
        results = {}
        
        results["per_species_average_precision_all"] = np.zeros(
            len(self.taxa), dtype=np.float32
        )
        loc_feat = enc.encode(self.obs_locs)

        # get classes to eval
        classes_of_interest = np.zeros(
            len(self.taxa), dtype=np.int64
        )
        for tt_id, tt in tqdm(enumerate(self.taxa), dynamic_ncols=True, total=len(self.taxa)):
            class_of_interest = np.where(
                np.array(self.class_to_taxa) == tt
            )[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = class_of_interest
        
        loc_emb = model.get_loc_emb(loc_feat)
        wt = model.model.layers[5].weights[0].numpy()[:, classes_of_interest]
        pred_mtx = tf.matmul(
            wt,
            loc_emb,
            transpose_a=True,
            transpose_b=True
        ).numpy()

        for tt_id, tt in tqdm(enumerate(self.taxa), dynamic_ncols=True, total=len(self.taxa)):
            class_of_interest = np.where(
                np.array(self.class_to_taxa) == tt
            )[0]
            if len(class_of_interest) == 0:
                results["per_species_average_precision_all"][tt_id] = np.nan
            else:
                pred = pred_mtx[tt_id,:]
                gt = np.zeros(self.obs_locs.shape[0], dtype=np.float32)
                gt[self.data["taxa_presence"][str(tt)]] = 1.0
                results["per_species_average_precision_all"][tt_id] = average_precision_score(gt, pred)
    
        valid_taxa = ~np.isnan(
            results["per_species_average_precision_all"]
        )
        per_species_average_precision_valid = results[
            "per_species_average_precision_all"
        ][valid_taxa]

        results["mean_average_precision"] = per_species_average_precision_valid.mean()
        results["num_eval_species_w_valid_ap"] = valid_taxa.sum()
        results["num_eval_species_total"] = len(self.taxa)

        return results

    def report(self, results):
        report_fields = [
            "mean_average_precision",
            "num_eval_species_w_valid_ap",
            "num_eval_species_total",
        ]
        for field in report_fields:
            print(f"{field}: {results[field]}")
