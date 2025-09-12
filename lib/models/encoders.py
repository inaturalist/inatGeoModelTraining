import math

import numpy as np
import tensorflow as tf


class CoordEncoder:
    def __init__(self, encoding_strategy, raster=None):
        self.encoding_strategy = encoding_strategy
        self.raster = raster

    def encode(self, locs, normalize=True):
        if normalize:
            locs = self._normalize_coords(locs)

        if self.encoding_strategy == "sinusoidal":
            loc_feats = self._encode_loc_sinusoidal(locs)
        elif self.encoding_strategy == "sinusoidal_hd":
            loc_feats = self._encode_loc_sinusoidal_hd(locs)

        if self.raster is not None:
            context_feats = self._bilinear_interpolate(locs, self.raster)
            loc_feats = np.concatenate((loc_feats, context_feats), 1)

        return loc_feats

    def num_input_feats(self):
        if self.encoding_strategy == "sinusoidal_hd":
            coord_feats = 14
        else:
            coord_feats = 4
        if self.raster is not None:
            return coord_feats + self.raster.shape[-1]
        else:
            return coord_feats

    def _normalize_coords(self, locs):
        # locs is in lon {-180, 180}, lat {-90, 90}
        # output is in the range [-1, 1]
        norm_locs = tf.stack(
            [
                locs[:, 0] / 180.0,
                locs[:, 1] / 90.0,
            ],
            axis=1,
        )
        return norm_locs

    def _encode_loc_sinusoidal(self, loc_ip, concat_dim=1):
        # assumes inputs location are in range -1 to 1
        # location is lon, lat
        feats = tf.concat(
            [tf.sin(loc_ip * math.pi), tf.cos(loc_ip * math.pi)], axis=concat_dim
        )
        return feats

    def _encode_loc_sinusoidal_hd(self, loc_ip, concat_dim=1):
        # assumes inputs location are in range -1 to 1
        # location is lon, lat

        lng = tf.expand_dims(loc_ip[:,0], 1)
        lat = tf.expand_dims(loc_ip[:,1], 1)

        feats = tf.concat([
            lng,
            lat,
            
            tf.sin(lng * math.pi),
            tf.cos(lng * math.pi), 
            tf.sin(lat * math.pi), 

            tf.sin(2 * lng * math.pi),
            tf.cos(2 * lng * math.pi), 
            tf.sin(2 * lat * math.pi), 

            tf.sin(4 * lng * math.pi),
            tf.cos(4 * lng * math.pi), 
            tf.sin(4 * lat * math.pi), 

            tf.sin(8 * lng * math.pi),
            tf.cos(8 * lng * math.pi), 
            tf.sin(8 * lat * math.pi), 
        ], axis=concat_dim)

        return feats

    def _bilinear_interpolate_new(self, loc_ip, data, remove_nans_raster=True):
        """
        Perform bilinear interpolation on a raster using normalized [-1, 1] lon/lat input.

        Args:
            loc_ip (tf.Tensor): [N x 2] tensor of [lon, lat] in [-1, 1] space.
            data (np.ndarray): [H x W x C] raster data.
            remove_nans_raster (bool): Whether to replace NaNs in `data` with 0.0.

        Returns:
            np.ndarray: [N x C] interpolated features for each location.
        """
        assert data is not None
        assert loc_ip.shape[1] == 2

        if remove_nans_raster:
            data = np.nan_to_num(data, nan=0.0)

        # Normalize from [-1, 1] to [0, 1]
        loc = (loc_ip + 1.0) / 2.0

        # Flip Y-axis for raster top-down layout
        x = loc[:, 0]
        y = 1.0 - loc[:, 1]

        # Convert to pixel indices
        px = x * (data.shape[1] - 1)
        py = y * (data.shape[0] - 1)

        # Corner integer indices
        x0 = tf.floor(px).numpy().astype(int)
        y0 = tf.floor(py).numpy().astype(int)
        x1 = np.clip(x0 + 1, 0, data.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, data.shape[0] - 1)

        # Deltas for interpolation
        dx = np.expand_dims(px - x0, axis=1)
        dy = np.expand_dims(py - y0, axis=1)

        # Fetch corner values
        top_left     = data[y0, x0, :]
        top_right    = data[y0, x1, :]
        bottom_left  = data[y1, x0, :]
        bottom_right = data[y1, x1, :]

        # Bilinear interpolation
        interp = (
            top_left * (1 - dx) * (1 - dy) +
            top_right * dx * (1 - dy) +
            bottom_left * (1 - dx) * dy +
            bottom_right * dx * dy
        )

        return interp

    def _bilinear_interpolate(self, loc_ip, data, remove_nans_raster=True):
        # this is almost entirely exactly from the sinr paper source
        # with some minor modifications for pytortch -> numpy/tf

        # loc is N x 2 vector, where each row is [lon,lat] entry
        #   each entry spans range [-1,1]
        # data is H x W x C, height x width x channel data matrix
        # op will be N x C matrix of interpolated features

        assert data is not None

        # map to [0,1], then scale to data size
        loc = (loc_ip + 1) / 2.0
        # latitude goes from +90 on top to bottom
        # longitude goes from -9i0 to 90 left to right
        loc = tf.stack(
            [
                loc[:, 0],
                1 - loc[:, 1],
            ],
            axis=1,
        )

        assert not np.any(np.isnan(loc))

        if remove_nans_raster:
            # replace with mean value (0 will be mean post-normalization)
            data[np.isnan(data)] = 0.0

        # cast locations into pixel space
        loc = tf.stack(
            [
                loc[:, 0] * (data.shape[1] - 1),
                loc[:, 1] * (data.shape[0] - 1),
            ],
            axis=1,
        )

        # integer pixel coordinates
        loc_int = np.floor(loc).astype(int)
        xx = loc_int[:, 0]
        yy = loc_int[:, 1]
        xx_plus = xx + 1
        xx_plus[xx_plus > (data.shape[1] - 1)] = data.shape[1] - 1
        yy_plus = yy + 1
        yy_plus[yy_plus > (data.shape[0] - 1)] = data.shape[0] - 1

        loc_delta = loc - np.floor(loc)
        dx = np.expand_dims(loc_delta[:, 0], 1)
        dy = np.expand_dims(loc_delta[:, 1], 1)

        interp_val = (
            data[yy, xx, :] * (1 - dx) * (1 - dy)
            + data[yy, xx_plus, :] * dx * (1 - dy)
            + data[yy_plus, xx, :] * (1 - dx) * dy
            + data[yy_plus, xx_plus, :] * dx * dy
        )

        return interp_val
