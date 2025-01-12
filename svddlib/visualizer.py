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
        if not hasattr(svdd_model, 'decision_function') or not hasattr(svdd_model, 'support_vectors_'):
             raise AttributeError("The SVDD model must implement `decision_function` and have `support_vectors_` attribute.")
        self.svdd_model = svdd_model        
        self.svdd_model
        self.resolution = resolution
        self.cmap       = cmap

    def _generate_grid(self, X):
        """
        Generate a grid for decision boundary visualization.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        xx : np.ndarray
            Meshgrid X-coordinates.
        yy : np.ndarray
            Meshgrid Y-coordinates.
        grid : np.ndarray
            Flattened grid for decision function computation
        """
        x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
        y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
        xx   , yy    = np.meshgrid(
            np.linspace(x_min, x_max, self.resolution),
            np.linspace(y_min, y_max, self.resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        return xx, yy, grid

    def plot_decision_boundary(self, X, y=None, margin=None, highlight_support_vectors=True, title=None):
        """
        Plots the decision boundary of the SVDD model.

        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, 2)
        y : np.ndarray, optional
            Labels for data points. If provided, points are colored by labels.
        margin : float, optional
            The margin width to visualize around the decision boundary.
        highlight_support_vectors : bool, optional (default=True)
            Whether to highlight support vectors in the plot.
        title : str, optional
            Title of plot.
        """
        if X.shape[1]!=2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Generate grid
        xx, yy, grid = self._generate_grid(X)

        # Calculate decision scores
        distances = self.svdd_model.decision_function(grid)
        decision_scores = distances.reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(10,8))
        plt.contour(xx, yy, decision_scores, cmap=self.cmap, alpha=0.8)
        plt.colorbar(label='Decision Score')

        # Plot margin if specified
        if margin is not None:
            plt.contour(xx, yy, decision_scores, levels=[-margin, 0, margin],
                        colors=['blue', 'black', 'red'], linestyles=['--', '-', '--'])

        # Plot data points
        if y is not None:
            plt.scatter(X[:,0], X[:,1], c=y, cmap=self.cmap, edgecolors='k')
        else:
            plt.scatter(X[:,0], X[:,1], color='black', edgecolors='k')
        
        # Highlight support vectors
        if highlight_support_vectors:
            support_vectors = self.svdd_model.support_vectors_
            plt.scatter(
                support_vectors[:, 0], support_vectors[:, 1],
                s=120, edgecolors='k', facecolor='none',
                label='Support Vectors', linewidths=2
            )

        # Add title and labels
        if title:
            plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid()
        plt.show()