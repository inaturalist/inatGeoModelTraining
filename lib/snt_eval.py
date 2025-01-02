import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tqdm.auto import tqdm

class EvaluatorSNT:
    def __init__(self, npy_path, split, val_frac, class_to_taxa):
        D = np.load(
            npy_path,
            allow_pickle=True
        )
        D = D.item()
        self.loc_indices_per_species = D["loc_indices_per_species"]
        self.labels_per_species = D["labels_per_species"]
        self.taxa = D["taxa"]
        self.obs_locs = D["obs_locs"]
        self.obs_locs_idx = D["obs_locs_idx"]

        self.class_to_taxa = class_to_taxa
        self.split = split
        self.val_frac = val_frac
        

    def run_eval(self, model, enc):
        results = {}
        results["per_species_average_precision_all"] = np.zeros(
            (len(self.taxa)), dtype=np.float32
        )
        
        loc_feat = enc.encode(self.obs_locs)
        classes_of_interest = np.zeros(
            len(self.taxa), dtype=np.int64
        )
        for tt_id, tt in tqdm(enumerate(self.taxa), total=len(self.taxa), dynamic_ncols=True):
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

        split_rng = np.random.default_rng()
        for tt_id, tt in tqdm(enumerate(self.taxa), total=len(self.taxa), dynamic_ncols=True):
            class_of_interest = np.where(
                np.array(self.class_to_taxa) == tt
            )[0]
            if len(class_of_interest) == 0:
                results["per_species_average_precision_all"][tt_id] = np.nan
            else:
                cur_loc_indices = np.array(self.loc_indices_per_species[tt_id])
                cur_labels = np.array(self.labels_per_species[tt_id])
                
                num_val = np.floor(len(cur_labels) * self.val_frac).astype(int)
                idx_rand = split_rng.permutation(len(cur_labels))
                # test set
                idx_sel = idx_rand[num_val:]
                cur_loc_indices = cur_loc_indices[idx_sel]
                cur_labels = cur_labels[idx_sel]

                pred = pred_mtx[tt_id, cur_loc_indices]
                results["per_species_average_precision_all"][tt_id] = average_precision_score(
                    (cur_labels > 0).astype(np.int32),
                    pred
                )
        
        valid_taxa = ~np.isnan(
            results["per_species_average_precision_all"]
        )
        per_species_average_precision_valid = results["per_species_average_precision_all"][valid_taxa]
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


