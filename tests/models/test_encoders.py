import numpy as np
import pytest

from lib.models import encoders

bioclim_raster = np.random.random((2160, 4320, 20)).astype(np.float32)
elev_raster = np.random.random((2160, 4320, 1)).astype(np.float32)

@pytest.mark.parametrize("strategy, raster, expected_feats", [
    ("sinusoidal", None, 4),
    ("sinusoidal_hd", None, 14),
    ("sinusoidal", bioclim_raster, 24),
    ("sinusoidal_hd", bioclim_raster, 34),
    ("sinusoidal", elev_raster, 5),
    ("sinusoidal_hd", elev_raster, 15),
])
def test_coord_encoder_input_feats(strategy, raster, expected_feats):
    encoder = encoders.CoordEncoder(strategy, raster)
    assert encoder.num_input_feats() == expected_feats


   
