# test_svdd.py
import numpy as np
import pytest
#from svdd import SVDD
from svddlib import SVDD
def test_get_params():
    # بررسی مقادیر پیش‌فرض پارامترها
    model = SVDD(C=0.5, kernel='poly', gamma='scale', degree=3, coef0=1.0, tol=1e-4, verbose=False)
    params = model.get_params()

    assert params['C'] == 0.5
    assert params['kernel'] == 'poly'
    assert params['gamma'] == 'scale'
    assert params['degree'] == 3
    assert params['coef0'] == 1.0
    assert params['tol'] == 1e-4
    assert params['verbose'] is False

def test_set_params():
    # بررسی تغییر پارامترها
    model = SVDD()
    model.set_params(C=1.0, kernel='linear', gamma='auto', degree=2, coef0=0.5, tol=1e-3, verbose=True)

    assert model.C == 1.0
    assert model.kernel == 'linear'
    assert model.gamma == 'auto'
    assert model.degree == 2
    assert model.coef0 == 0.5
    assert model.tol == 1e-3
    assert model.verbose is True

def test_fit():
    # بررسی متد fit
    X = np.array([[1, 2], [2, 3], [3, 4], [8, 9]])
    model = SVDD(C=0.9, kernel='linear', gamma='scale', verbose=False)

    model.fit(X)

    assert len(model.support_vectors_) > 0
    assert model.radius_ > 0
    assert model.center_ is not None

def test_predict():
    # بررسی پیش‌بینی داده‌ها
    X = np.array([[1, 2], [2, 3], [3, 4], [8, 9]])
    model = SVDD(C=0.9, kernel='linear', gamma='scale', verbose=False)

    model.fit(X)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({1, -1})

def test_decision_function():
    """
    تست متد decision_function برای اطمینان از محاسبات فاصله‌ها.
    """
    # داده‌های نمونه
    X = np.array([[1, 2], [2, 3], [3, 4], [8, 9]])
    model = SVDD(C=0.9, kernel='linear', gamma='scale', verbose=False)

    # اجرا و محاسبه فاصله‌ها
    model.fit(X)
    distances = model.decision_function(X)

    # بررسی تعداد و مقادیر فاصله‌ها
    assert len(distances) == len(X)
    assert np.all(np.isfinite(distances))
    assert all(isinstance(d, (float, int)) for d in distances)
    assert np.min(distances) >= -1e-3, f"Unexpected negative distance: {np.min(distances)}"

