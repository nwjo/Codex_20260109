import numpy as np

from one_channel.config import load_config
from one_channel.solver import OneChannelSolver


def test_short_transient_runs():
    config = load_config()
    config.solver["nx"] = 15
    config.solver["dt"] = 0.1
    config.solver["t_final"] = 0.2
    solver = OneChannelSolver(config)
    result = solver.run()
    assert not np.isnan(result.t_s).any()
    assert not np.isnan(result.t_g).any()
    assert np.all(result.c_g >= 0.0)
    assert np.all(result.c_g <= 1.0)
