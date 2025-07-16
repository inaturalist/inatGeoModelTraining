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

def test_coord_encoder_sinusoidal_expected_results():
    test_coords = np.array([
        [0.01, 90.0],           # north pole
        [0.01, -90.0],          # south pole
        [-122.4194, 37.7749],   # sf
        [0.1276, 51.5072],      # london
        [151.2057, -33.8727],   # sydney
        [28.1914, -25.7566],    # pretoria
        [-135.7681, 35.0116],   # kyoto
        [78.5141, -0.2233],     # quito
    ])

    expected_outputs = np.array([
        [ 1.74532924e-04, 1.22464680e-16, 9.99999985e-01,-1.00000000e+00],
        [ 1.74532924e-04,-1.22464680e-16, 9.99999985e-01,-1.00000000e+00],
        [-8.44146449e-01, 9.68364898e-01,-5.36112649e-01, 2.49538421e-01],
        [ 2.22703828e-03, 9.74313498e-01, 9.99997520e-01,-2.25195933e-01],
        [ 4.81666493e-01,-9.25510101e-01,-8.76354602e-01, 3.78722923e-01],
        [ 4.72418477e-01,-7.82751553e-01, 8.81374371e-01, 6.22334320e-01],
        [-6.97564142e-01, 9.39831033e-01,-7.16522343e-01, 3.41639618e-01],
        [ 9.79973738e-01,-7.79456151e-03, 1.99126777e-01, 9.99969622e-01],
    ])

    encoder = encoders.CoordEncoder("sinusoidal", None)
    encoded_coords = encoder.encode(test_coords)
    assert np.allclose(encoded_coords, expected_outputs)


