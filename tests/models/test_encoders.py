import pytest

from lib.models import encoders

@pytest.mark.parametrize("strategy, expected_feats", [
    ("sinusoidal", 4),
    ("sinusoidal_hd", 14),
])
def test_coord_encoder_input_feats(strategy, expected_feats):
    encoder = encoders.CoordEncoder(strategy)
    assert encoder.num_input_feats() == expected_feats


   
