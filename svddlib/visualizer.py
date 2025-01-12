# svddlib/visualizer.py

"""
Visualizer for SVDD Decision Boundary.

This module contains the `DecisionBoundaryVisualizer` class
for visualizing the decision boundary of an SVDD model.
"""
import numpy as np
import matplotlib.pyplot as plt

class DecisionBoundaryVisualizer:
    """
    Visualizes the decision boundary for an SVDD model.

    Parameters
    ----------
    svdd_model : SVDD
        The trained SVDD model.
    resolution : int, optional (default=100)
        The resolution of the visualization grid.
    cmap : str, optional (default='coolwram')
        The colormap to use for visualizing decision boundaries.
    """

    def __init__(self, svdd_model, resolution=100, cmap='coolwarm'):
        self.svdd_model = svdd_model
        self.resolution = resolution
        self.cmap       = cmap