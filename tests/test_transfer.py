import numpy as np

from one_channel.transport import nusselt_numbers, sherwood_numbers


def test_transfer_numbers_increase_with_re():
    re = np.array([100.0, 500.0])
    pr = np.array([0.7, 0.7])
    sc = np.array([0.9, 0.9])
    dh = 1.0e-3
    x = np.array([1.0e-3, 1.0e-3])
    nu = nusselt_numbers(re, pr, dh, x, nu1=3.0, combiner="max", power=4.0)
    sh = sherwood_numbers(re, sc, dh, x, sh1=3.0, combiner="max", power=4.0)
    assert nu[1] >= nu[0]
    assert sh[1] >= sh[0]
