import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from itertools import combinations, count, product, cycle
from collections import OrderedDict
from disjoint_set import DisjointSet

counter = count()
def uid(): return next(counter)


def sigma(T, X): return sum(1 for S in T if X.issubset(S))
def s(T, X): return sigma(T, X) / len(T)
def c(T, X, Y): return s(T, X.union(Y)) / s(T, X)


def flatten(xs):
    return [x for x, count in xs for _ in range(count)]


# Covariance and standard deviation.
def cov(x, y): return 1 / (len(x) - 1) * \
    sum(np.subtract(x, np.mean(x)) * np.subtract(y, np.mean(y)))


def stddev(x): return np.sqrt(cov(x, x))

# Similarity measures


def corr(x, y): return cov(x, y) / (stddev(x) * stddev(y))
def euclidean(x, y): return np.linalg.norm(np.subtract(x, y))


def cosine(x, y): return np.array(x).dot(np.array(y)) / \
    (np.linalg.norm(x) * np.linalg.norm(y))


def jaccard(x, y): return sum(np.logical_and(np.equal(x, 1), np.equal(
    y, 1)) / (sum(np.logical_or(np.not_equal(x, 0), np.not_equal(y, 0)))))
