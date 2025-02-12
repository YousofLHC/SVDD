import numpy as np
from sklearn.datasets        import make_blobs
from sklearn.model_selection import train_test_split
from svddlib                 import SVDD
from svddlib                 import DecisionBoundaryVisualizer


# Generate synthetic data
X, y = make_blobs(n_samples=10, centers=1, cluster_std=1, random_state=42)
y    = np.ones(X.shape[0]) # Labels for SVDD (all inlier)


# Split data into train and test sets
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = np.array([
    [0, 0],
    [1, 1],
    [-1, 1],
    [0.5, 0.5],
    [0.5, 0.1],
    [0.1, 0.3],
])
X_test = np.array([
    [1, -1],
    [0.2, 0.3],
    [-.5, -.5],
])

# Initialize and fit the SVDD model
svdd = SVDD(C=1.0, kernel='rbf', gamma='scale', tol=1e-4, verbose=False)
svdd.fit(X_train)

# Predict on the test set
y_pred = svdd.predict(X_test)
print(f"Number of inliers: {np.sum(y_pred==1)}")
print(f"Number of outliers: {np.sum(y_pred==-1)}")


# Visualize decision boundary
visualizer = DecisionBoundaryVisualizer(svdd_model=svdd)
visualizer.plot_decision_boundary(np.vstack([X_train,X_test]), title="SVDD Decision Boundary on `blobs` Synthetic Data.")

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from svddlib import SVDD
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_blobs(n_samples=10, centers=1, cluster_std=1, random_state=42)
y = np.ones(X.shape[0])  # Labels for SVDD (all inliers)

# Split data into train and test sets
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = np.array([
    [0, 0],
    [1, 1],
    [-1, 1],
    [0.5, 0.5],
    [0.5, 0.1],
    [0.1, 0.3],
])
X_test = np.array([
    [1, -1],
    [0.2, 0.3],
    [-.5, -.5],
])

# Initialize and fit the SVDD model
svdd = SVDD(C=1.0, kernel='rbf', gamma='scale', tol=1e-4, verbose=False)
svdd.fit(X_train)

# Predict on the test set
y_pred = svdd.predict(X_test)
print(f"Number of inliers: {np.sum(y_pred == 1)}")
print(f"Number of outliers: {np.sum(y_pred == -1)}")

# Create a grid of points for decision boundary visualization
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100),
    np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# Compute the decision function for the grid
decision_values = svdd.decision_function(grid).reshape(xx.shape)

# Plot decision boundary using sklearn's DecisionBoundaryDisplay
DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=decision_values).plot()
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=50, label="Training Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', s=50, label="Test Data")
plt.title("SVDD Decision Boundary on Blobs Synthetic Data")
plt.legend()
plt.show()
