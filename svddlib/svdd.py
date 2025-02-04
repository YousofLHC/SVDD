from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array
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
    - Implements support for `GridSearchCV` and `Pipeline` by providing compatible `get_params` and `set_params` methods.
    - Designed to work seamlessly with Scikit-learn's tools and standards for hyperparameter tuning and model chaining.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 degree=3, coef0=0.0, tol=1e-6, verbose=False):
        
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
        # Validate and preprocess inputs
        X, y = self._check_X_y(X,y) # * X_(n,m),y=None --> X_(n,m), y of ones_(n,)
        self.X = X
        self.y = y
        # Validate the input parameters
        self._validate_params()

        # Compute gamma if necessary
        self.gamma_ = self._compute_gamma(X)

        # Compute the kernel matrix
        self.K_ = self._compute_kernel(X) # * self.K_ = k(X,X)

        # Solve optimization problem 
        # * Pass k(x,x) of shape (n,n) and y of ones (n,)
        alphas = self._solve_optimization(self.K_, y)
        self.alpha = alphas
        # Identify support vectors
        self.support_         = np.where(alphas > self.tol)[0]
        self.support_vectors_ = X[self.support_]
        self.dual_coef_       = alphas[self.support_]

        # Compute center of the hypersphere
        #self.center_ = np.dot(self.dual_coef_, self.support_vectors_)
        self.center_ = np.dot(self.alpha.T, self.X)
        self.offset  = np.sum(np.multiply(np.dot(self.alpha, self.alpha.T), self.K_))
        # Compute the radius of the hypersphere
        tmp_5 = np.dot(np.ones((self.X.shape[0],1)), self.alpha.reshape(1,-1))
        tmp_6 = np.multiply(tmp_5, self.K_)
        tmp_  = -2*np.sum(tmp_6, axis=1, keepdims=True)
        self.radius_ = np.sqrt(
            np.mean(np.diag(self.K_)[self.support_])
            + self.offset
            + np.mean(tmp_[self.support_, 0])
        )
        # Select only relevant kernel values (support vectors to support vectors)
        distances = np.dot(self.K_[self.support_][:, self.support_], self.dual_coef_)
        self.radius_ = np.max(distances)

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
    
    def _compute_kernel(self, X, Y=None):
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
        if Y is None:
            Y = X # Default to set-similarity if Y is not provided
        if callable(self.kernel):
            # If a custom kernel function is provided
            return self.kernel(X, Y)

        # Define parameters specific to the selected kernel
        params = {}
        if self.kernel in {'poly', 'rbf', 'sigmoid'}:
            params['gamma'] = self.gamma_
        if self.kernel == 'poly':
            params['degree']=self.degree
            params['coef0']=self.coef0
        if self.kernel == 'sigmoid':
            params['coef0']=self.coef0

        # Compute the kernel matrix
        return pairwise_kernels(X, Y, metric=self.kernel, **params)

    def _check_X_y(self, X, y=None):
        """
        Validate and preprocess input data and  labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : None or np.ndarray of shape (n_samples, )
            Target labels. If None, it's assumed that SVDD is unsuprevised.
        
        Returns
        -------
        X : np.ndarray
            Validate and preprocessed input data.
        y : np.ndarray or None
            Validate target labels, or None if not provided
        """
        # Check if y is provided
        if y is None:
            y = np.ones(X.shape[0]) # Default to all 1s for unsupervised learning
        
        # Use check_X_y to validate and standardize inputs
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True,
                         dtype=np.float64, ensure_min_samples=2)

        # Check that labels are binary (-1 or 1)
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [1,-1])):
            raise ValueError(f"Labels must be binary (-1 or 1). Got {unique_labels}")
        
        return X, y
    
    def _solve_optimization(self, K, y):
        """
        Solve the dual optimization problem using cvxopt.

        Parameters
        ----------
        K : ndarray of shape (n_sample, n_sample)
            The kernel matrix
        y : ndarray of shape (n_sample, )
            Labels or weights for the data points.
            but included for future flexibility (e.g, hybrid SVDD variants)
        
        Returns
        -------
        alphas : ndarray of shape (n_sample, )
            The solution of the optimization problem
        """
        from cvxopt import matrix, solvers

        n_samples = K.shape[0]

        # Constructing the quadratic optimization problem
        P = matrix(K + K.T) # Symmetric kernel matrix
        # ? check if q is correct? q==diag(K)?
        q = matrix(-np.ones((n_samples, 1))) # Linear term
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)])) # Inequality constraints
        h = matrix(np.hstack( [np.zeros(n_samples), np.ones(n_samples)*self.C] )) # Bounds
        A = matrix(np.ones((1, n_samples))) # Equality constraint
        b = matrix(1.0) # Equality constraint value

        # Solve the quadratic program
        solvers.options['show_progress'] = self.verbose
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract the Lagrange multipliers (alphas)
        alphas = np.ravel(solution['x'])
        alphas[alphas < self.tol] = 0 # Set very small alphas to zero
        return alphas

    def decision_function(self, X):
        """
        Compute the distance of each sample to the hypersphere boundary.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Distances of each sample to the hypersphere boundary.
        """
        # Compute distance from the center
        # compute the distance between the samples and the center
        K = self._compute_kernel(X, self.X)
        K_ = self._compute_kernel(X, X)
        tmp_1 = np.dot(np.ones((X.shape[0], 1), dtype=np.int64), self.alpha.reshape(-1,1).T)
        tmp_2 = np.multiply(tmp_1, K)
        tmp_ = -2*np.sum(tmp_2, axis=1, keepdims=True)  
        distance = np.sqrt(np.mat(np.diag(K_)).T+self.offset+tmp_)
        distance = np.asarray(distance)
        print("Distances:", distance)
        print("Type:", type(distance))
        print("Shape:", np.shape(distance))
        return distance.ravel()
        # TODO Kernelized the distance 290 
        distances = self._compute_kernel(X-self.center_)-self.radius_ #np.linalg.norm(X - self.center_, axis=1)**2 - self.radius_**2 
            # Ensure no negative distances
        # ? Decision based on self.radius - distance 314
        distances = np.maximum(distances, 0)   
        print("Distances:", distances)
        print("Type:", type(distances))
        print("Shape:", np.shape(distances))

        return distances

    def predict(self, X):
        """
        Predict whether a sample is an inlier or outlier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predictions: 1 for inliers, -1 for outliers.
        """
        # Validate 
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        # Use decision_function to classify points
        distances = self.decision_function(X)
        predictions = np.where(distances >= 0, 1, -1)
        return predictions
    
    def fit_predict(self, X, y = None):
        """
        Fit the model using the training data and return predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored, SVDD is unsupervised

        Returns
        -------
        labels : ndarray of shape (n_samples, )
            Predicted labels: 1 for inliers, -1 for outliers
        """
        # Validate an preprocess inputs
        X,y = self._check_X_y(X, y)
        self.fit(X,y)
        return self.predict(X)
    def get_params(self, deep = True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and 
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'C'       : self.C,
            'kernel'  : self.kernel,
            'gamma'   : self.gamma,
            'degree'  : self.degree,
            'coef0'   : self.coef0,
            'tol'     : self.tol,
            'verbose' : self.verbose
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"""Invalid parameter '{key}' for estimator SVDD.
                                  Check the list of available parameters.
                                  with `estimator.get_params().keys()`.""")
        
        return self
    



