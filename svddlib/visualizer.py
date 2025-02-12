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

    def plot_decision_boundary(self, X, y=None, expand_ratio=0.2, n_grids=50,
                               color_map='RdYlBu', n_level=6, margin=None,
                               highlight_support_vectors=True, title=None):
        """
        Plots the decision boundary of the SVDD model.

        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, 2)
        y : np.ndarray, optional
            Labels for data points. If provided, points are colored by labels.
        expand_ratio : float, optional (default=0.2)
            The margin width to visualize around the decision boundary.
        n_grids : int, optional (default=50)
            Number of grid points in each dimension.
        color_map : str, optional (default='RdYlBu')
            Colormap for the decision boundary.
        n_level : int, optional (default=6)
            Number of levels in the contour plot.
        margin : float, optional
            Additional margin width for decision boundaries. #**redundant
        highlight_support_vectors : bool, optional (default=True)
            Whether to highlight support vectors in the plot.
        title : str, optional
            Title of plot.
        """
        if X.shape[1]!=2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Compute boundary range with expand_ratio
        x_range = np.zeros(shape=(n_grids, 2))
        for i in range(2):
            delta = (np.max(X[:, i]) - np.min(X[:, i]))*expand_ratio
            x_range[:, i] = np.linspace(
                np.min(X[:, i]) - delta,
                np.max(X[:, i]) + delta,
                n_grids
            )
        # Generate grid
        xx, yy, grid = self._generate_grid(X)
        #xx, yy = np.meshgrid(x_range[:, 0], x_range[:, 1])
        #grid   = np.c_[xx.ravel(), yy.ravel()]

        # Calculate decision scores
        distances = self.svdd_model.decision_function(grid)
        decision_scores =  distances.reshape(xx.shape) - self.svdd_model.radius_

        # Plot decision boundary
        plt.figure(figsize=(10,8))
        contour = plt.contourf(xx, yy, decision_scores, levels=n_level, cmap=color_map, alpha=0.8)
        plt.colorbar(contour, label='Decision Score')

        # Plot data points
        if y is not None:
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map, edgecolors='k', zorder=2)
        else:
            plt.scatter(X[:, 0], X[:, 1], c='k', edgecolors='k', zorder=2)
        # Plot margin if specified
        if margin is not None:
            plt.contour(xx, yy, decision_scores, levels=[-margin, 0, margin],
                        colors=['blue', 'black', 'red'], linestyles=['--', '-', '--'])

        
        # Highlight support vectors
        if highlight_support_vectors:
            support_vectors = self.svdd_model.support_vectors_
            plt.scatter(
                support_vectors[:, 0], support_vectors[:, 1],
                s=120, edgecolors='k', facecolor='none',
                label='Support Vectors', linewidths=2,
                zorder=3
            )

        # Add title and labels
        if title:
            plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid()
        plt.show()