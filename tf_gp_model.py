import math
import os

import numpy as np
import tensorflow as tf

from lib.geo_model_net import ResLayer

class TFGeoPriorModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"ResLayer": ResLayer},
            compile=False
        )

    def get_loc_emb(self, loc_feat):
        loc_emb = self.model.layers[0](loc_feat)
        x = self.model.layers[1](loc_emb)
        x = self.model.layers[2](x)
        x = self.model.layers[3](x)
        x = self.model.layers[4](x)
        return x

    def predict(self, loc_feat):
        return self.model(loc_feat)



