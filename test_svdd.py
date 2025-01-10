import numpy as np
from svdd import SVDD

def test_get_params():
    """
    Test the get_params method to ensure it returns the correct parameters.
    """
    model = SVDD(C=0.5, kernel='poly', gamma='scale', degree=4, coef0=1.0, tol=1e-4, verbose=True)
    params = model.get_params()

    assert params['C']      == 0.5
    assert params['kernel'] == 'poly'
    assert params['gamma']  == 'scale'
    assert params['degree'] == 4
    assert params['coef0']  == 1.0
    assert params['tol']    == 1e-4
    assert params['verbose'] is True

def test_set_params():
    """
    Test the set_params method to ensure it correctly updates parameters.
    """
    model = SVDD()
    model.set_params(C=0.7, kernel='linear', gamma='auto', degree=2, coef0=0.5, tol=1e-3, verbose=False)

    assert model.C      == 0.7
    assert model.kernel == 'linear'
    assert model.gamma  == 'auto'
    assert model.degree == 2
    assert model.coef0  == 0.5
    assert model.tol    == 1e-3
    assert model.verbose is False