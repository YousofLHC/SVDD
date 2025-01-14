import numpy as np
from sklearn.datasets        import make_blobs
from sklearn.model_selection import train_test_split
from svddlib                 import SVDD
from svddlib                 import DecisionBoundaryVisualizer


# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
y    = np.ones(X.shape[0]) # Labels for SVDD (all inlier)


# Split data into train and test sets
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize and fit the SVDD model
svdd = SVDD(C=1.0, kernel='rbf', gamma='scale', tol=1e-4, verbose=False)
svdd.fit(X_train)

# Predict on the test set
y_pred = svdd.predict(X_test)
print(f"Number of inliers: {np.sum(y_pred==1)}")
print(f"Number of outliers: {np.sum(y_pred==-1)}")


# Visualize decision boundary
visualizer = DecisionBoundaryVisualizer(svdd_model=svdd)
visualizer.plot_decision_boundary(X_train, title="SVDD Decision Boundary on `blobs` Synthetic Data.")