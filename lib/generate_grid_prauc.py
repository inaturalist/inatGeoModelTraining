import math
import os
from pathlib import Path

import tifffile
import geopandas as gpd
import pandas as pd
import numpy as np
import h3pandas
from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import wandb

class TaxonRangeEvalCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        taxon_range_recalls_file,
        taxon_range_csvs,
        tax_file,
        elevation_file,
        sample_size=100,
        h3_resolution=4,
    ):
        super(TaxonRangeEvalCallback, self).__init__()

        self.taxon_range_recalls = pd.read_csv(taxon_range_recalls_file)
        self.taxon_range_csvs = taxon_range_csvs
        self.tax = pd.read_csv(tax_file)
        self.gdfk = self.generate_df_from_elevation_tif(elevation_file, h3_resolution)
        self.taxon_ids = sorted(
            list(self.tax[~self.tax.leaf_class_id.isna()].taxon_id.values)
        )
        self.sample_taxon_ids = self.taxon_ids[:sample_size]

    def get_sample_prauc(self, model):
        feats = self.features_for_one_class_elevation(
            latitude=list(self.gdfk.lat),
            longitude=list(self.gdfk.lng),
            elevation=list(self.gdfk.elevation),
        )
        praucs = []

        for taxon_id in self.sample_taxon_ids:
            prauc = self.process_prauc(taxon_id, feats)

            if prauc is not None:
                praucs.append(prauc)

        if len(praucs) > 0:
            return np.mean(praucs)
        else:
            return 0.0

    def on_epoch_end(self, epoch, logs=None):
        sample_prauc = self.get_sample_prauc(model=self.model)
        wandb.log({"epoch": epoch, "sample_prauc": sample_prauc}, commit=False)

    def generate_df_from_elevation_tif(self, elevation_file, h3_resolution):
        im = tifffile.imread(elevation_file)

        im_df = pd.DataFrame(im)
        im_df.index = np.linspace(90, -90, 2160)
        im_df.columns = np.linspace(-180, 180, 4320)
        im_df = im_df.reset_index()
        im_df = im_df.melt(
            id_vars=["index"],
        )
        im_df.columns = ["lat", "lng", "elevation"]

        print("processing h3 and making features...")
        elev_dfh3 = im_df.h3.geo_to_h3(h3_resolution)
        elev_dfh3 = (
            elev_dfh3.drop(columns=["lng", "lat"])
            .groupby("h3_0" + str(h3_resolution))
            .mean()
        )
        gdfk = elev_dfh3.h3.h3_to_geo()
        gdfk["lng"] = gdfk["geometry"].x
        gdfk["lat"] = gdfk["geometry"].y
        _ = gdfk.pop("geometry")
        gdfk = gdfk.rename_axis("h3index")

        return gdfk

    def eval_one_class_elevation_from_features(self, features, class_of_interest):
        return tf.keras.activations.sigmoid(
            tf.matmul(
                tf.expand_dims(
                    self.model.layers[5].weights[0][:, class_of_interest], axis=0
                ),
                features,
                transpose_b=True,
            )
        ).numpy()

    def features_for_one_class_elevation(self, latitude, longitude, elevation):
        encoded_loc = TaxonRangeEvalCallback.encode_loc(latitude, longitude, elevation)
        loc_emb = self.model.layers[0](encoded_loc)
        x = self.model.layers[1](loc_emb)
        x = self.model.layers[2](x)
        x = self.model.layers[3](x)
        x = self.model.layers[4](x)
        return x

    @staticmethod
    def encode_loc(latitude, longitude, elevation):
        latitude = np.array(latitude)
        longitude = np.array(longitude)
        elevation = np.array(elevation)
        elevation = elevation.astype("float32")
        grid_lon = longitude.astype("float32") / 180.0
        grid_lat = latitude.astype("float32") / 90.0

        elevation[elevation > 0] = elevation[elevation > 0] / 6574.0
        elevation[elevation < 0] = elevation[elevation < 0] / 32678.0
        norm_elev = elevation

        if np.isscalar(grid_lon):
            grid_lon = np.array([grid_lon])
        if np.isscalar(grid_lat):
            grid_lat = np.array([grid_lat])
        if np.isscalar(norm_elev):
            norm_elev = np.array([norm_elev])
        norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

        encoded_loc = tf.concat(
            [
                tf.sin(norm_loc * math.pi),
                tf.cos(norm_loc * math.pi),
                tf.expand_dims(norm_elev, axis=1),
            ],
            axis=1,
        )

        return encoded_loc

    def process_prauc(self, taxon_id, feats):
        if (
            self.taxon_range_recalls[
                self.taxon_range_recalls.taxon_id.eq(taxon_id)
            ].shape[0]
            == 0
        ):
            return None

        if self.taxon_range_recalls[
            (self.taxon_range_recalls["taxon_id"] == taxon_id)
            & (self.taxon_range_recalls["recall"] > 0.9)
        ].empty:
            return None

        tri = self.taxon_range_csvs / f"{taxon_id}.csv"
        if not os.path.exists(tri):
            return None

        taxon_range_index = pd.read_csv(tri, header=None)
        taxon_range_index.rename(columns={0: "h3index_new"}, inplace=True)
        tr_h3 = self.gdfk.loc[self.gdfk.index.isin(taxon_range_index.h3index_new)]

        try:
            class_of_interest = self.tax[self.tax.taxon_id == taxon_id].iloc[0][
                "spatial_class_id"
            ]
        except Exception:
            print(f"{taxon_id} is not in the taxonomy")
            return None

        try:
            preds = self.eval_one_class_elevation_from_features(
                feats, class_of_interest
            )
        except Exception as e:
            print(f"can't eval one class {taxon_id}")
            return None

        self.gdfk["pred"] = tf.squeeze(preds).numpy()

        auc_presences = self.gdfk[self.gdfk.index.isin(tr_h3.index)]["pred"]
        auc_absences = self.gdfk[~self.gdfk.index.isin(tr_h3.index)]["pred"]
        test = np.concatenate(([1] * len(auc_presences), [0] * len(auc_absences)))
        predictions = np.concatenate((auc_presences, auc_absences))
        precision, recall, thresholds = precision_recall_curve(test, predictions)
        return auc(recall, precision)
