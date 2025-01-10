from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np

class SVDD(BaseEstimator, OutlierMixin):
    """
    Support Vector Data Description (SVDD) for anomaly detection.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Must be strictly positive.
    kernel : {'linear', 'ploy', 'rbf', 'sigmoid'}, callable, default='rbf'
        Kernel function to compute similarity between data points.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
    coef0 : float, default=0.0
        Independent term in kernel function. Used olny in 'poly' and 'sigmoid'.
    tol : float, default=1e-6
        Tolerance for determining support vectors.
    verbose : bool, default=False
        Whether to print detailed progress.

    Notes
    -----
    Why use BaseEstimator and OutlierMixin?
    - **BaseEstimator**:
        - Automatically provides `get_params` and `set_params` for parameter management.
        - Ensure compatibility with Scikit-learn tools like `Pipeline` and `GridSearchCV`.
        - Reduces repetitive code by managing parameters automatically.
    - **OutlierMixin**:
        - Adds specific functionality for anomaly detection (e.g., `fit_predict`).
        - Identifies the class as an anomaly detection model within Scikit-learn's ecosystem.
        - Simplifies integration with Scikit-learn's evaluation tools.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, coef0=0.0, tol=1e-6, verbose=False):
        self.C       = C
        self.kernel  = kernel
        self.gamma   = gamma
        self.degree  = degree
        self.coef0   = coef0
        self.tol     = tol
        self.verbose = verbose
    def _validate_params(self):
        """
        Validate the input parameters of the SVDD model.

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        # C must be strictly positive
        if not isinstance(self.C, (int, float)) or self.C <= 0:
            raise ValueError("C must be strictly positive float. Got {0}".format(self.C))
        
        # Gamma must be a positive float or one of {'scale', 'auto'}
        if isinstance(self.gamma, (int, float)) and self.gamma <= 0:
            raise ValueError("Gamma must be a positive float, 'scale', or 'auto'. Got {0}".format(self.gamma))
        elif isinstance(self.gamma, str) and self.gamma not in {'scale', 'auto'}:
            raise ValueError(f"Gamma must be 'scale', 'auto', or a positive float. Got {self.gamma}")
        
        # Kernel must be a valid option or callable
        if not (self.kernel in {'linear', 'poly', 'rbf', 'sigmoid'} or callable(self.kernel)):
            raise ValueError(f"Kernel must be one of {'linear', 'poly', 'rbf', 'sigmoid'} or a callable function. Got {self.kernel}")
        
        # Degree must be a positive integer (if relevent)
        if not isinstance(self.degree, int) or self.degree <=0 :
            raise ValueError(f"Degree must be a positive integer. Got {self.degree}")
        
        # Tolerance must be positive
        if not isinstance(self.tol, (int, float)) or self.tol <= 0:
            raise ValueError(f"Tolerance (tol) must be a strictly positive float. Got {self.tol}")
    
    def fit(self, X, y=None):
        """
        Fit the SVDD Model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored; SVDD is unsupervised.

        Returns
        -------
        self : object
            Returns the instance itself
        """

        # Validate the input parameters
        self._validate_params()

        # Compute gamma if necessary
        self.gamma = self._compute_gamma(X)

        # Compute the kernel matrix
        self.K_ = self._compute_kernel(X)

        # TODO: Add optimization logic here
        return self

    def _compute_gamma(self, X):
        """
        Compute the gamma value based on the input data if gamma is 'scale' or 'auto'.

        Parameters
        ----------
        X : ndarray of shape (n_sample, n_feature)
            The input data.
        
        Returns
        -------
        gamma_value : float
            The computed value.
        """
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                return 1.0/(X.shape[1]*X.var()) if X.var() != 0 else 1.0
            elif self.gamma == 'auto':
                return 1.0/X.shape[1]
            else:
                raise ValueError(f"Gamma must be 'scale', 'auto', or a positive float. Got {self.gamma}")
        return self.gamma
    
    def _compute_kernel(self, X, Y):
        """
        Compute kernel matrix using the specified kernel.

        Parameters
        ----------
        X : ndarray of shape (n_sample_X, n_feature)
            First input dataset
        Y : ndarray of shape (n_sample_Y, n_feature), optional
            Second input dataset. If None, Y is set to X.

        Returns
        -------
        K : ndarray of shape (n_sample_X, n_sample_Y)
            The computed kernel matrix
        """
        if callable(self.kernel):
            # If a custom kernel function is provided
            return self.kernel(X, Y)
        else:
            # Use Scikit-learn's pairwise_kernels for predefined kernels
            return pairwise_kernels(X, Y, metric=self.kernel, gamma=self.gamma,
                                    degree=self.degree, coef0=self.coef0)

    def _solve_optimization(self, K, y):
        """
        Solve the dual optimization problem using cvxopt.

        Parameters
        ----------
        K : ndarray of shape (n_sample, n_sample)
            The kernel matrix
        y : ndarray of shape (n_sample, )
            Labels or weights for the data points.
        
        Returns
        -------
        alphas : ndarray of shape (n_sample, )
            The solution of the optimization problem
        """
        from cvxopt import matrix, solvers

        n_samples = K.shape[0]

        # Constructing the quadratic optimization problem
        P = matrix(K + K.T) # Symmetric kernel matrix
        q = matrix(-np.ones(n_samples, 1)) # Linear term
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)])) # Inequality constraints
        h = matrix(np.hstack( [np.zeros(n_samples), np.ones(n_samples), self.C] )) # Bounds
        A = matrix(np.ones(1, n_samples)) # Equality constraint
        b = matrix(1.0) # Equality constraint value

        # Solve the quadratic program
        solvers.options['show_progress'] = self.verbose
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract the Lagrange multipliers (alphas)
        alphas = np.ravel(solution['x'])
        return alphas

