import numpy as np

from one_channel.solver import thomas_solver


def test_thomas_solver_matches_numpy():
    n = 6
    lower = -1.0 * np.ones(n - 1)
    main = 4.0 * np.ones(n)
    upper = -1.0 * np.ones(n - 1)
    d = np.arange(1, n + 1, dtype=float)

    mat = np.diag(main) + np.diag(upper, k=1) + np.diag(lower, k=-1)
    expected = np.linalg.solve(mat, d)
    computed = thomas_solver(lower, main, upper, d)
    assert np.allclose(computed, expected)
