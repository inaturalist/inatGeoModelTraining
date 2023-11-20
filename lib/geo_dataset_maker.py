import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

from pathlib import Path


class DiscretizedInatGeoModelDataset:
    def __init__(self, config):
        self.config = config

    def _load_spatial_dataset(self, path):
        return pd.read_csv(
            path,
            usecols=[
                "latitude",
                "longitude",
                "spatial_class_id",
                "taxon_id",
                "community",
                "captive",
            ],
            dtype={
                "latitude": float,
                "longitude": float,
                "spatial_class_id": int,
                "taxon_id": int,
                "community": bool,
                "captive": bool,
            },
        )

    def _clean_spatial_dataset(self):
        print("  remove data at inner nodes")
        self.spatial_train = self.spatial_train[
            self.spatial_train.taxon_id.isin(self.leaf_tax.taxon_id.unique())
        ]

        if self.config["train_only_cid_data"]:
            print("  dropping data without cid")
            self.spatial_train = self.spatial_train[
                self.spatial_train.community == True
            ]

        if self.config["train_only_wild_data"]:
            print("  dropping data that's captive/cultivated")
            self.spatial_train = self.spatial_train[self.spatial_train.captive == False]

        # rename columns for h3pandas
        self.spatial_train = self.spatial_train.rename(
            {"latitude": "lat", "longitude": "lng"}, axis=1
        )

        # drop locations outside of valid range
        num_obs = self.spatial_train.shape[0]
        self.spatial_train = self.spatial_train[
            (
                (self.spatial_train["lat"] <= 90)
                & (self.spatial_train["lat"] >= -90)
                & (self.spatial_train["lng"] <= 180)
                & (self.spatial_train["lng"] >= -180)
            )
        ]
        if (num_obs - self.spatial_train.shape[0]) > 0:
            print(
                " ",
                num_obs - self.spatial_train.shape[0],
                "items filtered due to invalid locations",
            )

        # drop null island locations
        num_obs = self.spatial_train.shape[0]
        self.spatial_train = self.spatial_train[
            ((self.spatial_train["lat"] != 0) & (self.spatial_train["lng"] != 0))
        ]
        if (num_obs - self.spatial_train.shape[0]) > 0:
            print(
                " ",
                num_obs - self.spatial_train.shape[0],
                "observation(s) with a 0 lat and lng entry out of",
                num_obs,
                "removed",
            )

        self.spatial_train = self.spatial_train[["lat", "lng", "spatial_class_id"]]

    def _convert_to_discretized_h3(self):
        print("  adding h3 index")
        h3_resolution = self.config.get("h3_resolution")
        assert h3_resolution != None, "Must define h3 resolution for training"

        h3_index_name = "h3_0{}".format(h3_resolution)
        self.spatial_train_h3 = self.spatial_train.h3.geo_to_h3(h3_resolution)

        print("  doing h3 pivot")
        self.spatial_train_h3_dense = self.spatial_train_h3.pivot_table(
            index=h3_index_name, values="spatial_class_id", aggfunc=set
        )

        self.spatial_train_h3_dense[
            "spatial_class_id"
        ] = self.spatial_train_h3_dense.spatial_class_id.apply(lambda x: list(x))

    def _make_random_samples(self):
        def make_samples(batch_size):
            rand_loc = np.random.uniform(size=(batch_size, 2))

            theta1 = 2.0 * math.pi * rand_loc[:, 0]
            theta2 = np.arccos(2.0 * rand_loc[:, 1] - 1.0)

            lat = 1.0 - 2.0 * theta2 / math.pi
            lng = (theta1 / math.pi) - 1.0

            return list(zip(lng, lat))

        samples = make_samples(self.config["num_random_samples"])

        scaled_samples = []
        for sample in samples:
            lng, lat = sample
            lng = lng * 180
            lat = lat * 90
            scaled_samples.append((lng, lat))

        negatives_df = pd.DataFrame(scaled_samples, columns=["lng", "lat"])
        negatives_h3 = negatives_df.h3.geo_to_h3(self.config["h3_resolution"])

        self.negatives_h3 = negatives_h3[~negatives_h3.index.duplicated(keep="first")]

    def _combine_spatial_data_with_empties(self):
        empty_valid_indices = list(
            set(self.negatives_h3.index).difference(
                set(self.spatial_train_h3_dense.index)
            )
        )
        negative_valid_h3 = self.negatives_h3[
            self.negatives_h3.index.isin(empty_valid_indices)
        ]

        combined = pd.concat([negative_valid_h3, self.spatial_train_h3_dense])
        combined["spatial_class_id"] = combined["spatial_class_id"].apply(
            lambda d: d if isinstance(d, list) else []
        )
        combined = combined[["spatial_class_id"]]
        combined = combined.h3.h3_to_geo()
        combined = combined.sort_index()
        combined["lng"] = combined["geometry"].x
        combined["lat"] = combined["geometry"].y
        _ = combined.pop("geometry")

        self.combined = combined

    def _combine_spatial_data_elevation(self):
        print("  loading elevation")
        elevations = pd.read_csv(
            self.config["elevation_file"], index_col=self.combined.index.name
        )
        above_sea_level_norm = elevations.elevation[elevations.elevation > 0] / np.max(
            elevations.elevation[elevations.elevation > 0]
        )
        below_sea_level_norm = (
            elevations.elevation[elevations.elevation < 0]
            / np.min(elevations.elevation[elevations.elevation < 0])
        ) * -1
        elev_normalized = pd.concat([below_sea_level_norm, above_sea_level_norm])
        elev_norm_df = pd.DataFrame(
            {
                "elevation": elev_normalized,
            }
        )

        print("  merging elevation")
        self.combined_with_elevation = pd.merge(
            self.combined,
            elev_norm_df,
            left_index=True,
            right_index=True,
        )

    def _make_tfrecords(self):
        def create_tf_example(l0, l1, l2, l3, elevation, leaf_class_ids):
            tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "l0": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[l0])
                        ),
                        "l1": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[l1])
                        ),
                        "l2": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[l2])
                        ),
                        "l3": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[l3])
                        ),
                        "elevation": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[elevation])
                        ),
                        "leaf_class_ids": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=leaf_class_ids)
                        ),
                    }
                )
            )
            return tf_example

        print("  writing tfrecords")
        with tf.io.TFRecordWriter(self.tfrecord_file) as writer:
            total = len(self.combined_with_elevation)
            i = 0
            for _, row in self.combined_with_elevation.iterrows():
                if i % 100_000 == 0:
                    print("  {} of {} written".format(i, total))
                i += 1

                lat = row["lat"]
                lng = row["lng"]

                norm_lat = lat / 90.0
                norm_lng = lng / 180.0
                norm_loc = np.array([norm_lng, norm_lat])
                encoded_loc = np.concatenate(
                    [np.sin(norm_loc * math.pi), np.cos(norm_loc * math.pi)]
                )

                l0 = encoded_loc[0]
                l1 = encoded_loc[1]
                l2 = encoded_loc[2]
                l3 = encoded_loc[3]

                elevation = row["elevation"]
                cids = row["spatial_class_id"]
                example = create_tf_example(l0, l1, l2, l3, elevation, cids)
                writer.write(example.SerializeToString())
            writer.close()

    def make_dataset(self):
        print("loading taxonomy")
        taxonomy_path = Path("/") / self.config["export_dir"] / "taxonomy.csv"
        self.tax = pd.read_csv(taxonomy_path)
        self.leaf_tax = self.tax[~self.tax.leaf_class_id.isna()]
        self.num_leaf_taxa = len(self.leaf_tax)

        print("loading spatial training dataset")
        spatial_data_path = Path("/") / self.config["export_dir"] / "spatial_data.csv"
        self.spatial_train = self._load_spatial_dataset(spatial_data_path)

        print("cleaning spatial training dataset")
        spatial_train = self._clean_spatial_dataset()

        # should have no nas
        assert self.spatial_train.isna().any().any() == False
        # don't train if cleaning the dataframe changed the number of available taxa
        assert len(self.spatial_train.spatial_class_id.unique()) == self.num_leaf_taxa

        print("doing h3 conversion and discretizing dataframe")
        self._convert_to_discretized_h3()

        print("making random samples for negatives")
        self._make_random_samples()
        print("combine spatial data with negatives")
        self.spatial_train_h3_dense_with_empties = (
            self._combine_spatial_data_with_empties()
        )

        # print("layering in elevation")
        self._combine_spatial_data_elevation()

        # still no nas
        assert self.combined_with_elevation.isna().any().any() == False

        # write the tfrecords
        assert self.config["export_dir"] != None, "we need a valid export dir"
        assert self.config["export_dir"] != "", "we need a valid export dir"

        os.makedirs(
            os.path.join(self.config["export_dir"], "geo_spatial_grid_datasets"),
            exist_ok=True,
        )
        self.tfrecord_file = os.path.join(
            self.config["export_dir"],
            "geo_spatial_grid_datasets",
            "r{}_empty_cells_with_elevation.tf".format(self.config["h3_resolution"]),
        )

        if self.config["full_shuffle_before_tfrecords"]:
            self.config("shuffle all data before making tfrecords")
            self.combined_with_elevation = self.combined_with_elevation.sample(frac=1)

        print("make tfrecords")
        self._make_tfrecords()
