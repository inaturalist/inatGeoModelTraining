from collections import defaultdict

import h3pandas
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

from .data_cleaner import clean_dataset

def _get_idx_subsample_observations_h3(labels, h3_idx, hard_cap=-1):
    if hard_cap == -1:
        return np.arange(len(labels))

    print(f"  subsampling (up to) {hard_cap} per class for the training set")
    print("   using h3 index to constrain subsampling to at most 1 observation per cell")
    class_counts = {id: 0 for id in np.unique(labels)}
    cells_dict = defaultdict(list)
    
    ss_rng = np.random.default_rng()
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in tqdm(idx_rand):
        class_id = labels[i]
        h3_cell = h3_idx[i]
        if class_counts[class_id] < hard_cap:
            if class_id not in cells_dict[h3_cell]:
                idx_ss.append(i)
                class_counts[class_id] += 1
                cells_dict[h3_cell].append(class_id)
    idx_ss = np.sort(idx_ss)
    print(f"  final training set size: {len(idx_ss)}")
    return idx_ss


def _get_idx_subsample_observations(labels, hard_cap=-1):
    if hard_cap == -1:
        return np.arange(len(labels))

    print(f"  subsampling (up to) {hard_cap} per class for the training set")
    class_counts = {id: 0 for id in np.unique(labels)}
    ss_rng = np.random.default_rng()
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in idx_rand:
        class_id = labels[i]
        if class_counts[class_id] < hard_cap:
            idx_ss.append(i)
            class_counts[class_id] += 1
    idx_ss = np.sort(idx_ss)
    print(f"  final training set size: {len(idx_ss)}")
    return idx_ss


def make_subsampled_dataset(
    hard_cap, encoded_locs, class_ids, batch_size, shuffle_buffer_size=None, h3_idx=None
):
    if h3_idx is not None:
        ss_idx = _get_idx_subsample_observations_h3(class_ids, h3_idx, hard_cap=hard_cap)
    else:
        ss_idx = _get_idx_subsample_observations(class_ids, hard_cap=hard_cap)

    locs_ss = np.array(encoded_locs)[ss_idx]
    class_ids_ss = np.array(class_ids)[ss_idx]
    num_train_steps_per_epoch = locs_ss.shape[0] // batch_size
    ds = tf.data.Dataset.from_tensor_slices((locs_ss, class_ids_ss))

    if shuffle_buffer_size is None:
        ds = ds.shuffle(buffer_size=ds.cardinality())
    else:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, num_train_steps_per_epoch

def load_inat_dataset_from_parquet_h3(spatial_data_file, h3_resolution):
    print("inat style dataset")
    print(" reading parquet")
    spatial_data = pd.read_parquet(
        spatial_data_file,
        columns=[
            "leaf_class_id",
            "latitude",
            "longitude",
            "taxon_id",
            "spatial_class_id",
        ],
    )
    spatial_data = spatial_data.dropna(subset="leaf_class_id")
    print(" cleaning dataset")
    spatial_data = clean_dataset(spatial_data)
    print(" shuffling")
    spatial_data = spatial_data.sample(frac=1)
    print(" extracting locs")
    locs = np.vstack(
        (spatial_data["longitude"].values, spatial_data["latitude"].values)
    ).T.astype(np.float32)

    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)
    unique_taxa, _ = np.unique(taxon_ids, return_inverse=True)

    print(" extracting spatial class ids")
    class_ids = spatial_data["spatial_class_id"].values.astype(int)
    print(f" found {len(unique_taxa)} unique taxa")

    print(" doing h3 conversion and extracting h3 index")
    spatial_data_h3 = spatial_data.h3.geo_to_h3(
        h3_resolution,
        lat_col="latitude",
        lng_col="longitude"
    )
    spatial_data_h3.reset_index(inplace=True)
    h3_index_name = f"h3_{h3_resolution:02}"
    h3_idx = spatial_data_h3[h3_index_name].values

    return locs, class_ids, unique_taxa, h3_idx



def load_inat_dataset_from_parquet(spatial_data_file, inner_nodes):
    print("inat style dataset")
    print(" reading parquet")
    spatial_data = pd.read_parquet(
        spatial_data_file,
        columns=[
            "leaf_class_id",
            "latitude",
            "longitude",
            "taxon_id",
            "spatial_class_id",
        ],
    )

    if not inner_nodes:
        spatial_data = spatial_data.dropna(subset="leaf_class_id")

    # we won't need this anymore
    _ = spatial_data.pop("leaf_class_id")

    print(" cleaning dataset")
    spatial_data = clean_dataset(spatial_data)
    print(" shuffling")
    spatial_data = spatial_data.sample(frac=1)
    print(" extracting locs")
    locs = np.vstack(
        (spatial_data["longitude"].values, spatial_data["latitude"].values)
    ).T.astype(np.float32)

    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)
    unique_taxa, _ = np.unique(taxon_ids, return_inverse=True)

    print(" extracting spatial class ids")
    class_ids = spatial_data["spatial_class_id"].values.astype(int)
    print(f" found {len(unique_taxa)} unique taxa")

    return locs, class_ids, unique_taxa

def load_sinr_dataset_from_parquet_h3(file, h3_resolution):
    print("sinr style dataset")
    print(" reading parquet")
    spatial_data = pd.read_parquet(
        file,
        columns=[
            "longitude",
            "latitude",
            "taxon_id",
        ],
    )
    spatial_data = clean_dataset(spatial_data)

    print(" extracting locs")
    locs = np.vstack(
        (spatial_data["longitude"].values, spatial_data["latitude"].values)
    ).T.astype(np.float32)

    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)

    print(" making class_ids")
    unique_taxa, class_ids = np.unique(taxon_ids, return_inverse=True)
    print(f" found {len(unique_taxa)} unique taxa")

    print(" doing h3 conversion and extracting h3 index")
    spatial_data_h3 = spatial_data.h3.geo_to_h3(
        h3_resolution,
        lat_col="latitude",
        lng_col="longitude"
    )
    spatial_data_h3.reset_index(inplace=True)
    h3_index_name = f"h3_{h3_resolution:02}"
    h3_idx = spatial_data_h3[h3_index_name].values

    return locs, class_ids, unique_taxa, h3_idx


def load_sinr_dataset_from_parquet(file):
    print("sinr style dataset")
    print(" reading parquet")
    spatial_data = pd.read_parquet(
        file,
        columns=[
            "longitude",
            "latitude",
            "taxon_id",
        ],
    )
    spatial_data = clean_dataset(spatial_data)

    print(" extracting locs")
    locs = np.vstack(
        (spatial_data["longitude"].values, spatial_data["latitude"].values)
    ).T.astype(np.float32)

    print(" extracting taxon_id")
    taxon_ids = spatial_data["taxon_id"].values.astype(int)

    print(" making class_ids")
    unique_taxa, class_ids = np.unique(taxon_ids, return_inverse=True)
    print(f" found {len(unique_taxa)} unique taxa")

    return locs, class_ids, unique_taxa
