import numpy as np


def cov(x, y):
    """Covariance"""
    return 1 / (len(x) - 1) * sum(np.subtract(x, np.mean(x)) * np.subtract(y, np.mean(y)))


def stddev(x):
    """Standard deviation"""
    return np.sqrt(cov(x, x))


def corr(x, y):
    """Correlation"""
    return cov(x, y) / (stddev(x) * stddev(y))


def euclidean(x, y):
    """Euclidean distance (i.e. L2-norm)"""
    return np.linalg.norm(np.subtract(x, y))


def cosine(x, y):
    """cos(theta) where theta is the angle between the two vectors x and y."""
    return np.array(x).dot(np.array(y)) / (np.linalg.norm(x) * np.linalg.norm(y))


def jaccard(x, y):
    """Jaccard"""
    return sum(np.logical_and(np.equal(x, 1), np.equal(
        y, 1)) / (sum(np.logical_or(np.not_equal(x, 0), np.not_equal(y, 0)))))
