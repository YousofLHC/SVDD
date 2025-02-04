# test_svdd.py
import numpy as np
import pytest
#from svdd import SVDD
from svddlib import SVDD
from svddlib import DecisionBoundaryVisualizer
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
    assert set(predictions.ravel()).issubset({1, -1})

#def test_decision_function():
#    """
#    تست متد decision_function برای اطمینان از محاسبات فاصله‌ها.
#    """
#    # داده‌های نمونه
#    X = np.array([[1, 2], [2, 3], [3, 4], [8, 9]])
#    model = SVDD(C=0.9, kernel='linear', gamma='scale', verbose=False)
#
#    # اجرا و محاسبه فاصله‌ها
#    model.fit(X)
#    distances = model.decision_function(X)
#
#    # بررسی تعداد و مقادیر فاصله‌ها
#    assert len(distances) == len(X)
#    assert np.all(np.isfinite(distances))
#    assert all(isinstance(d, (float, int)) for d in distances)
#    assert np.min(distances) >= -1e-3, f"Unexpected negative distance: {np.min(distances)}"
#
#def test_simple_2d_data():
#    """
#    Test SVDD with a simple 2D
#    """
#
#    # Create a simple 2D dataset
#    X = np.array([
#        [1, 1],
#        [2, 2],
#        [3, 3],
#        [4, 4],
#        [5, 5]
#    ])
#    y = np.array([1, 1, 1, -1, -1])
#
#    # Initialize and fit the SVDD model
#    model = SVDD(C=0.9, kernel='rbf', gamma='scale', verbose=False)
#    model.fit(X)
#
#    # Assertion for the model
#    assert model.radius_ > 0, "Radius should be greater than zero for simple 2D data."
#    assert model.center_ is not None, "Center should not be None after fitting."
#    assert len(model.support_vectors_) > 0, "Support vectors should be identified."
#
#    # Initialize the Visualizer
#    visualizer = DecisionBoundaryVisualizer(svdd_model=model)
#
#    # Plot decision boundary (no exception should occur)
#    try:
#        visualizer.plot_decision_boundary(X, y=y, title="Decision Boundary for simple 2D Data")
#    except Exception as e:
#        pytest.fail(f"Visualization faild for simple 2D data with error: {e}")
#
#def test_multidimensional_data():
#    """
#    Test SVDD with multi-dimensional data.
#    """
#
#    # Create multi-dimensional dataset
#    X = np.random.rand(100,5) # 100 samples with 5 features
#
#    # Initialize and fit the SVDD model
#    model = SVDD(C=1.0, kernel='rbf', gamma='scale', verbose=False)
#    model.fit(X)
#
#    # Assertion for the model
#    assert model.radius_ > 0, "Radius should be greater than zero for multi-dimensional data."
#    assert model.center_ is not None, "Center should not be None after fitting."
#    assert len(model.support_vectors_) > 0, "Support vectors should be identified."
#
#    # Test decision function
#    distances = model.decision_function(X)
#    assert distances.shape[0] == X.shape[0], "Distance output should match the number of samples."
#    assert np.all(np.isfinite(distances)), "All distances should be finite."
#
#    # Predict and verify labels
#    predictions = model.predict(X)
#    assert len(predictions)==len(X), "Predictions should match the number of samples."
#    assert set(predictions).issubset({1,-1}), "Predictions should be 1 (inliers) or -1 (outliers)."
#
#def test_mixed_2d_and_visualization():
#    """
#    Test SVDD with mixed multi-dimensional data and visualize a reduced 2D subset.
#    """
#
#    # Create a mixed dataset: 100 samples with 10 features
#    X_high_dim = np.random.rand(100,10)
#
#    # Select a 2D subset for visualization
#    X_2d = X_high_dim[:, :2]
#
#
#    # Initialize and fit the SVDD model
#    model = SVDD(C=1.0, kernel='rbf', gamma='scale', verbose=False)
#    model.fit(X_2d)#(X_high_dim)
#
#    # Assertion for the model
#    assert model.radius_ > 0, "Radius should be greater than zero for high-dimensional data."
#    assert model.center_ is not None, "Center should not be None after fitting."
#
#    # Visualize using only the 2D subset
#    visualizer = DecisionBoundaryVisualizer(svdd_model=model)
#    try:
#        visualizer.plot_decision_boundary(X_2d,title="Visualization for 2D Subset of High-Dimensional data.")
#    except Exception as e:
#        pytest.fail(f"Visualization failed for reduced 2D subset with error: {e}")