import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
import tifffile
from tqdm.auto import tqdm


def coord_grid(grid_size, split_ids=None, split_of_interest=None):
    # generate a grid of locations spaced evenly in coordinate space
    feats = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
    mg = np.meshgrid(
        np.linspace(-180, 180, feats.shape[1]), np.linspace(90, -90, feats.shape[0])
    )
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]
    if split_ids is None or split_of_interest is None:
        # return feats for all locs
        # this will be N x 2
        return feats.reshape(feats.shape[0] * feats.shape[1], 2)
    else:
        # only select a subset of locations
        # this will be N_subset x 2
        ind_y, ind_x = np.where(split_ids == split_of_interest)
        return feats[ind_y, ind_x, :]


def create_spatial_split(raster, mask, train_amt=1.0, cell_size=25):
    # creates a checkerboard style train test split
    # 0 is invalid, 1 is train, 2 is test
    # c_size is untits of pixels
    split_ids = np.ones((raster.shape[0], raster.shape[1]))
    start = cell_size
    for ii in np.arange(0, split_ids.shape[0], cell_size):
        if start == 0:
            start = cell_size
        else:
            start = 0
        for jj in np.arange(start, split_ids.shape[1], cell_size * 2):
            split_ids[ii : ii + cell_size, jj : jj + cell_size] = 2
    split_ids = split_ids * mask
    if train_amt < 1.0:
        # take a subset of the data
        tr_y, tr_x = np.where(split_ids == 1)
        inds = np.random.choice(
            len(tr_y), int(len(tr_y) * (1.0 - train_amt)), replace=False
        )
        split_ids[tr_y[inds], tr_x[inds]] = 0

    return split_ids


class EvaluatorGeoFeatures:
    def __init__(self, data_path, mask, class_to_taxa):
        self.data_path = data_path
        self.country_mask = tifffile.imread(mask) == 1
        self.raster_names = [
            "ABOVE_GROUND_CARBON",
            "ELEVATION",
            "LEAF_AREA_INDEX",
            "NON_TREE_VEGITATED",
            "NOT_VEGITATED",
            "POPULATION_DENSITY",
            "SNOW_COVER",
            "SOIL_MOISTURE",
            "TREE_COVER",
        ]
        self.raster_names_log_transform = ["POPULATION_DENSITY"]
        self.cell_size = 25
        self.class_to_taxa = class_to_taxa

    def load_raster(self, raster_name, log_transform=False):
        raster = tifffile.imread(
            os.path.join(self.data_path, raster_name + ".tif")
        ).astype(np.float32)
        valid_mask = ~np.isnan(raster).copy() & self.country_mask

        # log scaling
        if log_transform:
            raster[valid_mask] = np.log1p(raster[valid_mask] - raster[valid_mask].min())

        # 0/1 scaling:
        raster[valid_mask] -= raster[valid_mask].min()
        raster[valid_mask] /= raster[valid_mask].max()

        return raster, valid_mask

    def get_split_labels(self, raster, split_ids, split_of_interest):
        # get the GT lables for a subset
        inds_y, inds_x = np.where(split_ids == split_of_interest)
        return raster[inds_y, inds_x]

    def get_split_feats(self, model, enc, split_ids, split_of_interest):
        locs = coord_grid(
            self.country_mask.shape,
            split_ids=split_ids,
            split_of_interest=split_of_interest,
        )
        locs_enc = enc.encode(locs)
        loc_emb = model.get_loc_emb(locs_enc)
        return loc_emb

    def run_eval(self, model, enc):
        results = {}
        for raster_name in tqdm(self.raster_names):
            do_log_transform = raster_name in self.raster_names_log_transform
            raster, valid_mask = self.load_raster(raster_name, do_log_transform)
            split_ids = create_spatial_split(
                raster, valid_mask, cell_size=self.cell_size
            )
            feats_train = self.get_split_feats(
                model, enc, split_ids=split_ids, split_of_interest=1
            )
            feats_test = self.get_split_feats(
                model, enc, split_ids=split_ids, split_of_interest=2
            )
            labels_train = self.get_split_labels(raster, split_ids, 1)
            labels_test = self.get_split_labels(raster, split_ids, 2)
            scaler = MinMaxScaler()
            feats_train_scaled = scaler.fit_transform(feats_train)
            feats_test_scaled = scaler.transform(feats_test)
            clf = RidgeCV(
                alphas=(0.1, 1.0, 10.0), cv=10, fit_intercept=True, scoring="r2"
            ).fit(feats_train_scaled, labels_train)
            train_score = clf.score(feats_train_scaled, labels_train)
            test_score = clf.score(feats_test_scaled, labels_test)
            results[f"train_r2_{raster_name}"] = float(train_score)
            results[f"test_r2_{raster_name}"] = float(test_score)
            results[f"alpha_{raster_name}"] = float(clf.alpha_)
        return results

    def report(self, results):
        report_fields = [x for x in results if "test_r2" in x]
        for field in report_fields:
            print(f"{field}: {results[field]}")
        print(np.mean([results[field] for field in report_fields]))
