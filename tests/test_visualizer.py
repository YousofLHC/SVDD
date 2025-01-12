import pytest
import numpy as np
from svddlib import DecisionBoundaryVisualizer
from svddlib import SVDD

@pytest.fixture
def sample_data():
    X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 3.0],
    [8.0, 8.0]
    ])
    y = np.array([1,1,-1,-1])
    return X, y

@pytest.fixture
def trained_svdd(sample_data):
    X, y = sample_data
    model = SVDD(C=0.9, kernel='rbf', gamma='scale')
    model.fit(X)
    return model


def test_initialization(trained_svdd):
    """
    Test the initialization of the DecisionBoundaryVisualizer.
    """
    # Create Visualizer
    visualizer = DecisionBoundaryVisualizer(trained_svdd)

    # Check if the SVDD model is correctly assigned
    assert visualizer.svdd_model is trained_svdd, "SVDD model was not correctly assigned to the visualizer."

    # Check if the default resolution is set
    assert visualizer.resolution == 100, "Default resolution should be 100, but got a different value."

    # Check if the default colormap is set
    assert visualizer.cmap =='coolwarm', "Default colormap should be 'coolwarm', but got a different value."

def test_plot_decision_boundary():
    """
    Test the `plot_decision_boundary` method to ensure it visualizes decision boundaries correctly. 
    """

    # Generate 2D random data
    np.random.seed(42)
    X = np.random.rand(100,2)*10 # Generate 100 data points in [0, 10]
    y = np.random.choice([1,-1], size=100) # Generate random labels

    # Train SVDD model
    model = SVDD(C=0.9, kernel='linear', gamma='scale')
    model.fit(X)

    # Create DecisionBoundaryVisualizer Object
    visualizer = DecisionBoundaryVisualizer(svdd_model=model)

    # Call plot_decision_boundary
    try:
        visualizer.plot_decision_boundary(X, y, title="Test Decision Boundary")
    except Exception as e:
        assert False, f"`plot_decision_boundary` raised an exception: {e}"

    # Specify margin
    try:
        visualizer.plot_decision_boundary(X, y, margin=0.5, title='Test with margin')
    except Exception as e:
        assert False, f"`plot_decision_boundary` with margin raised an exception: {e}"

    # Call without labels
    try:
        visualizer.plot_decision_boundary(X, title='Test without Labels')
    except Exception as e:
        assert False, f"`plot_decision_boundary` without labels raised an exception: {e}"