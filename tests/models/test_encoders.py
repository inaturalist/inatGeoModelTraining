import numpy as np
import pytest

from lib.models import encoders

# stubs
bioclim_raster = np.random.random((2160, 4320, 20)).astype(np.float32)
elev_raster = np.random.random((2160, 4320, 1)).astype(np.float32)

# fixtures
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


def test_coord_encoder_sinusoidal_hd_expected_results():
    expected_outputs = np.array([
         [ 1.00000000e+00,  5.55555556e-05,  1.22464680e-16, -1.00000000e+00,
           1.74532924e-04, -2.44929360e-16,  1.00000000e+00,  3.49065843e-04,
          -4.89858720e-16,  1.00000000e+00,  6.98131644e-04, -9.79717439e-16,
           1.00000000e+00,  1.39626295e-03],
         [-1.00000000e+00,  5.55555556e-05, -1.22464680e-16, -1.00000000e+00,
           1.74532924e-04,  2.44929360e-16,  1.00000000e+00,  3.49065843e-04,
           4.89858720e-16,  1.00000000e+00,  6.98131644e-04,  9.79717439e-16,
           1.00000000e+00,  1.39626295e-03],
         [ 4.19721111e-01, -6.80107778e-01,  9.68364898e-01,  2.49538421e-01,
          -8.44146449e-01,  4.83288495e-01, -8.75461153e-01,  9.05115177e-01,
          -8.46200606e-01,  5.32864461e-01, -7.69649225e-01, -9.01820460e-01,
          -4.32110933e-01,  9.82791216e-01],
         [ 5.72302222e-01,  7.08888889e-04,  9.74313498e-01, -2.25195933e-01,
           2.22703828e-03, -4.38822875e-01, -8.98573583e-01,  4.45406552e-03,
           7.88629286e-01,  6.14868970e-01,  8.90804268e-03,  9.69807353e-01,
          -2.43872301e-01,  1.78153785e-02],
         [-3.76363333e-01,  8.40031667e-01, -9.25510101e-01,  3.78722923e-01,
           4.81666493e-01, -7.01023782e-01, -7.13137895e-01, -8.44221297e-01,
           9.99853248e-01,  1.71313139e-02, -9.04996413e-01,  3.42575996e-02,
          -9.99413036e-01,  7.70005692e-01],
         [-2.86184444e-01,  1.56618889e-01, -7.82751553e-01,  6.22334320e-01,
           4.72418477e-01, -9.74266311e-01, -2.25399988e-01,  8.32755077e-01,
           4.39199230e-01, -8.98389691e-01,  9.22095647e-01, -7.89144120e-01,
           6.14208073e-01, -7.13632018e-01],
         [ 3.89017778e-01, -7.54267222e-01,  9.39831033e-01,  3.41639618e-01,
          -6.97564142e-01,  6.42167031e-01, -7.66564742e-01,  9.99640587e-01,
          -9.84525209e-01,  1.75243008e-01,  5.35978008e-02, -3.45062319e-01,
          -9.38579776e-01, -1.07041519e-01],
         [-2.48111111e-03,  4.36189444e-01, -7.79456151e-03,  9.99969622e-01,
           9.79973738e-01, -1.55886495e-02,  9.99878490e-01,  3.90278024e-01,
          -3.11735105e-02,  9.99513988e-01, -7.18655654e-01, -6.23167197e-02,
           9.98056424e-01, -9.99457598e-01]
    ])

    encoder_hd = encoders.CoordEncoder("sinusoidal_hd", None)
    encoded_coords = encoder_hd.encode(test_coords)
    assert np.allclose(encoded_coords, expected_outputs)

   

