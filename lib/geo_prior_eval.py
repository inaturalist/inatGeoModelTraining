import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class EvaluatorGeoPrior:
    def __init__(self, preds_path, meta_path, batch_size, class_to_taxa):
        # load vision model preds
        self.data = np.load(preds_path)
        print(f"{self.data['probs'].shape[0]} total test obs")

        # load locations
        meta = pd.read_csv(meta_path)
        self.obs_locs = np.vstack(
            (meta["longitude"].values, meta["latitude"].values)
        ).T.astype(np.float32)

        self.class_to_taxa = class_to_taxa

        # taxonomic mapping
        self.taxon_map = self.find_mapping_between_models(
            self.data["model_to_taxa"], self.class_to_taxa
        )
        print(
            f"{self.taxon_map.shape[0]} out of {len(self.data['model_to_taxa'])} taxa in both vision and geo models"
        )
        self.batch_size = batch_size

    def find_mapping_between_models(self, vision_taxa, geo_taxa):
        taxon_map = np.ones((vision_taxa.shape[0], 2), dtype=np.int32) * -1
        taxon_map[:, 0] = np.arange(vision_taxa.shape[0])
        geo_taxa_arr = np.array(geo_taxa)
        for tt_id, tt in tqdm(
            enumerate(vision_taxa), dynamic_ncols=True, total=len(vision_taxa)
        ):
            ind = np.where(geo_taxa_arr == tt)[0]
            if len(ind) > 0:
                taxon_map[tt_id, 1] = ind[0]
        inds = np.where(taxon_map[:, 1] > -1)[0]
        taxon_map = taxon_map[inds, :]
        return taxon_map

    def convert_to_inat_vision_order(
        self, geo_pred_ip, vision_top_k_prob, vision_top_k_inds, vision_taxa, taxon_map
    ):
        vision_pred = np.zeros(
            (geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32
        )
        geo_pred = np.ones((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        vision_pred[
            np.arange(vision_pred.shape[0])[..., np.newaxis], vision_top_k_inds
        ] = vision_top_k_prob

        geo_pred[:, taxon_map[:, 0]] = geo_pred_ip[:, taxon_map[:, 1]]
        return geo_pred, vision_pred

    def run_eval(self, model, enc):

        results = {}

        batch_start = np.hstack(
            (
                np.arange(0, self.data["probs"].shape[0], self.batch_size),
                self.data["probs"].shape[0],
            )
        )
        correct_pred = np.zeros(self.data["probs"].shape[0])

        for bb_id, bb in tqdm(
            enumerate(range(len(batch_start) - 1)),
            dynamic_ncols=True,
            total=len(batch_start) - 1,
        ):
            batch_inds = np.arange(batch_start[bb], batch_start[bb + 1])

            vision_probs = self.data["probs"][batch_inds, :]
            vision_inds = self.data["inds"][batch_inds, :]
            gt = self.data["labels"][batch_inds]

            obs_locs_batch = self.obs_locs[batch_inds, :]
            loc_feat = enc.encode(obs_locs_batch)
            geo_pred = model.predict(loc_feat).numpy()

            geo_pred, vision_pred = self.convert_to_inat_vision_order(
                geo_pred,
                vision_probs,
                vision_inds,
                self.data["model_to_taxa"],
                self.taxon_map,
            )

            comb_pred = np.argmax(vision_pred * geo_pred, 1)
            comb_pred = comb_pred == gt
            correct_pred[batch_inds] = comb_pred

        results["vision_only_top_1"] = float(
            (self.data["inds"][:, -1] == self.data["labels"]).mean()
        )
        results["vision_geo_top_1"] = float(correct_pred.mean())
        return results

    def report(self, results):
        vision_only = round(results["vision_only_top_1"], 3)
        vision_plus_geo = round(results["vision_geo_top_1"], 3)
        gain = round(results["vision_geo_top_1"] - results["vision_only_top_1"], 3)

        print(f"Overall acc vision-only: {vision_only}")
        print(f"Overall acc w geo model: {vision_plus_geo}")
        print(f"Gain: {gain}")
