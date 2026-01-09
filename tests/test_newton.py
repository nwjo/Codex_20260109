import numpy as np

from one_channel.solver import damped_newton


def test_damped_newton_converges():
    def func(x):
        return np.array([
            x[0] ** 2 + x[1] - 2.0,
            x[0] + x[1] ** 2 - 2.0,
        ])

    x0 = np.array([1.0, 1.0])
    sol, iters = damped_newton(func, x0, tol=1e-10, max_iter=50, damping=1.0, bounds=(0.0, 2.0))
    assert iters < 20
    assert np.allclose(func(sol), 0.0, atol=1e-8)
