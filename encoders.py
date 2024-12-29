import math

import tensorflow as tf

class CoordEncoder:
    def encode(self, locs, normalize=True):
        if normalize:
            norm_locs = self.normalize_coords(locs)
            return self.encode_loc_sinusoidal(norm_locs)
        else:
            return self.encode_loc_sinusoidal(locs)

    def normalize_coords(self, locs):
        # locs is in lon {-180, 180}, lat {-90, 90}
        # output is in the range [-1, 1]
        norm_locs = locs.copy()
        norm_locs[:,0] /= 180.0
        norm_locs[:,1] /= 90.0
        return norm_locs

    def encode_loc_sinusoidal(self, loc_ip, concat_dim=1):
        # assumes inputs location are in range -1 to 1
        # location is lon, lat
        feats = tf.concat(
            [tf.sin(loc_ip*math.pi), tf.cos(loc_ip*math.pi)],
            axis=concat_dim
        )
        return feats
