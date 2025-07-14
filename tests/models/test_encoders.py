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
def test_coord_encoder_input_num_feats(strategy, raster, expected_feats):
    encoder = encoders.CoordEncoder(strategy, raster)
    assert encoder.num_input_feats() == expected_feats


def test_coord_encoder_hd_fix_polar_continuitiy():
    north_pole_coords = np.array([[0.01, 90.0]])
    south_pole_coords = np.array([[0.01, -90.0]])

    # start by confirming the polar continuity with non-hd inputs
    encoder = encoders.CoordEncoder("sinusoidal", None)
    encoded_north_pole = encoder.encode(north_pole_coords)
    encoded_south_pole = encoder.encode(south_pole_coords)

    assert np.allclose(
        encoded_north_pole, 
        encoded_south_pole
    )

    # now confirm that there is no polar continuity with hd inputs
    hd_encoder = encoders.CoordEncoder("sinusoidal_hd", None)
    hd_encoded_north_pole = hd_encoder.encode(north_pole_coords)
    hd_encoded_south_pole = hd_encoder.encode(south_pole_coords)
    assert not np.allclose(
        hd_encoded_north_pole,
        hd_encoded_south_pole
    )

    


