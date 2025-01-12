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

    def plot_decision_boundary(self, X, y=None, title=None):
        """
        Plots the decision boundary of the SVDD model.

        Parameters
        ----------
        X : ndarray
            Input data (n_samples, 2)
        y: ndarray, optional
            Labels for data points. If provided, points are colored by labels.
        title : str, optional
            Title of plot.
        """
        if X.shape[1]!=2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Generate grid
        x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
        y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
        xx   , yy    = np.meshgrid(
            np.linspace(x_min, x_max, self.resolution),
            np.linspace(y_min, y_max, self.resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Calculate decision scores
        distances = self.svdd_model.decision_function(grid)
        decision_scores = distances.reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(10,8))
        plt.contour(xx, yy, decision_scores, cmap=self.cmap, alpha=0.8)
        plt.colorbar(label='Decision Score')

        # Plot data points
        if y is None:
            plt.scatter(X[:,0], X[:,1], c=y, cmap=self.cmap, edgecolors='k')
        else:
            plt.scatter(X[:,0], X[:,1], color='black', edgecolors='k')
        
        # Add title and labels
        if title:
            plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid()
        plt.show()