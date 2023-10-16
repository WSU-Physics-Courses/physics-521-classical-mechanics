"""Basic Tests for Assignment 0.
"""
import numpy as np

# import pytest

from phys_581_2021 import assignment0


def test_arrays():
    """Make sure code works with arrays."""
    np.random.seed(3)
    a, b, c = np.random.random((3, 4)) - 0.5
    x1, x2 = assignment0.quadratic_equation(a=a, b=b, c=c)
    for x in (x1, x2):
        res = x * (a * x + b) + c
        assert np.allclose(res, 0)
