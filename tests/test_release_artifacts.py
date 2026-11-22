"""The released reliability model and per-clip results."""
import os

import numpy as np
import pytest

from ps4d.io import read_results
from ps4d.measurements import FEATURES
from ps4d.metrics import complete, median
from ps4d.reliability import VarianceModel
from ps4d.utils import REPO_ROOT, RELIABILITY_SHA256, default_reliability_model, sha256_file

FROZEN = ["ps4d_pi3x", "lift_pi3x", "ps", "branch4d_pi3x", "spatracker2", "lift_vda", "lift_moge2", "joint_graph",
          "disagreement_weights"]


def test_reliability_model_checksum_and_prediction():
    assert sha256_file(default_reliability_model()) == RELIABILITY_SHA256
    model = VarianceModel.load()
    assert model.features == FEATURES
    meas = [{"dt": 3, "n_pts": 40, "vis_frac": 0.9, "spread_px": 60.0, "disp": 2.0, "iqr": 0.01, "n_pairs": 700,
             "is_pnp": 0},
            {"dt": 12, "n_pts": 8, "vis_frac": 0.2, "spread_px": 12.0, "disp": 9.0, "pnp_inl": 0.6, "pnp_rms": 2.5,
             "z_spread": 0.3, "n_pairs": 0, "is_pnp": 1}]
    v = model.variance(meas)
    assert np.all(np.isfinite(v)) and np.all(v > 0)
    assert v[1] > v[0]                       # few, crowded points with a poor rigid fit are less reliable


@pytest.mark.parametrize("name, expected", [("dexycb", 0.588), ("hot3d_quest3", 0.375), ("droid", 0.617)])
def test_results_reproduce_ps4d_median(name, expected):
    path = os.path.join(REPO_ROOT, "results", "test", f"{name}.json.gz")
    clips = complete(read_results(path)["clips"], FROZEN)
    assert round(median(clips, "ps4d_pi3x"), 3) == expected
