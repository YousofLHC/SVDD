import numpy as np
from sklearn.datasets        import make_moons
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing   import StandardScaler
from svddlib                 import SVDD, DecisionBoundaryVisualizer               


# Generate complex synthetic data (moons dataset)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
y    = np.ones(X.shape[0]) # SVDD works with only inliers by default


# Split data into train and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Define a pipline with preprocessing and the SVDD model
pipline = Pipeline([
    ('scaler', StandardScaler()), # Scale data
    ('svdd', SVDD()) # Add SVDD model
])

# Define the parameter grid for GridSearchCV\
param_grid = {
    'svdd__C' : [0.1, 1.0, 10.0], # Regularization parameter
    'svdd__kernel' : ['linear', 'rbf'], # Kernel type
    'svdd__gamma'  : ['scale', 'auto'], # Gamma for kernels
    'svdd__tol'    : [1e-4, 1e-6], # Tolerance
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=pipline, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y=np.ones(X_train.shape[0])) # All data points are inliers

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_pred     = best_model.predict(X_test)
print(f"Number of inliers in test set: {np.sum(y_pred==1)}")
print(f"Number of outliers in test set: {np.sum(y_pred==-1)}")

# Visualize the decision boundary
visualizer = DecisionBoundaryVisualizer(svdd_model=best_model.named_steps['svdd'])
visualizer.plot_decision_boundary(X_train, title="SVDD Decision Boundary on Moons Dataset")