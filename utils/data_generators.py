import numpy as np
from sklearn.datasets import make_classification, make_regression, make_moons

def generate_classification_data(n_samples, noise=0.3):
    """Generate synthetic data for binary classification"""
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X, y

def generate_regression_data(n_samples, noise=0.3):
    """Generate synthetic data for regression"""
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise)
    return X, y

def generate_nonlinear_data(n_samples):
    """Generate synthetic nonlinear data"""
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, n_samples)
    return X, y

def generate_svm_data(n_samples, noise=0.3):
    """Generate synthetic data for SVM classification"""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=1, 
                             n_clusters_per_class=1, class_sep=0.6)
    return X, y