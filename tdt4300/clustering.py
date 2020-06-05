import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from itertools import combinations, count, product, cycle
from collections import OrderedDict
from disjoint_set import DisjointSet


def plot_clusters(data, centroids):
    """
    Shows a scatter plot with the data points clustered according to the centroids.
    """
    # Assigning the data points to clusters/centroids.
    clusters = [[] for _ in range(centroids.shape[0])]
    for i in range(data.shape[0]):
        distances = np.linalg.norm(data[i] - centroids, axis=1)
        clusters[np.argmin(distances)].append(data[i])

    # Plotting clusters and centroids.
    fig, ax = plt.subplots()
    for c in range(centroids.shape[0]):
        if len(clusters[c]) > 0:
            cluster = np.array(clusters[c])
            ax.scatter(cluster[:, 0], cluster[:, 1], s=7)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')


def kmeans_step(data, centroids):
    """
    Function implementing the k-means clustering.

    :param data
        data
    :param centroids
        initial centroids
    :return
        final centroids
    """
    # data:  nx2
    # centroids: mx2
    clusters = np.zeros(len(data))
    for i, d in enumerate(data):
        clusters[i] = np.argmin(np.linalg.norm(d - centroids, axis=1))

    return np.array([
        data[clusters == i].mean(axis=0) for i in range(len(centroids))
    ])


def kmeans(data, centroids, iterations=-1):
    while iterations != 0:
        iterations -= 1
        plot_clusters(data, centroids)
        plt.show()
        new_centroids = kmeans_step(data,  centroids)
        if (new_centroids == centroids).all():
            break
        centroids = new_centroids

    return data, centroids


def silhouette_score(data, centroids):
    """
    Function implementing the k-means clustering.

    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    clusters = np.zeros(len(data))
    for i, d in enumerate(data):
        clusters[i] = np.argmin(np.linalg.norm(d - centroids, axis=1))

    a = np.zeros(len(data))
    b = np.zeros(len(data))
    for i, (d, c) in enumerate(zip(data, clusters)):
        select = np.logical_and(clusters == c, (data != d).all(axis=1))
        a[i] = np.mean(np.linalg.norm(d - data[select], axis=1))
        b[i] = np.min([
            np.mean(
                np.linalg.norm(d - data[clusters == j], axis=1)
            ) for j in range(len(centroids)) if j != c
        ])

    s = (b - a) / np.maximum(a, b)

    return np.mean(s)


def minlink(group1, group2):
    return min(np.linalg.norm(np.subtract(p, q)) for p, q in product(group1, group2))


def maxlink(group1, group2):
    return max(np.linalg.norm(np.subtract(p, q)) for p, q in product(group1, group2))


class Cluster:
    def __init__(self, items, subclusters=[], name=''):
        self.items = items
        self.subclusters = subclusters
        # Used to construct identifier. Assumed to be a single character,
        # and only to be used by the single-item clusters.
        self._name = name

    def union(self, other):
        return Cluster(
            items=[x for xs in [self.items, other.items] for x in xs],
            subclusters=[self, other],
        )

    def name(self):
        # We will cache the name after first access, since we might otherwise run into some accidental
        # exponential complexity using self.dot().
        self._name = self._name or ''.join(
            sorted(child.name() for child in self.subclusters))
        return self._name

    def dot(self, dot=None):
        """Creates a graph that can be visualized through Graphviz"""
        # This is just to create a default for the top-level caller, due to
        # Python's Mutable Default Argument we can't do dot = graphviz.Digraph()
        dot = dot or graphviz.Digraph()

        # To avoid dublicated calling of this.
        name = self.name()
        # Create a node for this cluster.
        dot.node(name)
        # Create nodes for each child, as well as an edge connecting
        # child -> parent.
        for child in self.subclusters:
            child.dot(dot)
            dot.edge(child.name(), name)

        return dot

    def _repr_svg_(self):
        return self.dot()._repr_svg_()


def hac(table, metric, print_progress=False):
    # Init to single-item clusters.
    clusters = [
        Cluster([point], name=name) for name, point in table.items()
    ]

    while len(clusters) > 1:
        # Number of clusters
        N = len(clusters)

        # Find the indices of the two closest groups
        (a, b) = min(
            ((i, j) for i in range(N) for j in range(i + 1, N)),
            key=lambda t: metric(clusters[t[0]].items, clusters[t[1]].items),
        )

        if print_progress:
            print(f'Merging ({clusters[a].name()}, {clusters[b].name()})')

        # Insert the union of the two close groups and the remove them.
        # NOTE: order of removal of a and b are important. a < b by construction,
        # so if we pop(a) first, we'll have shifted the element at the b index.
        clusters.append(clusters[a].union(clusters[b]))
        clusters.pop(b)
        clusters.pop(a)

    # We should now have a single cluster remaining
    return clusters[0]


# Constants to label a points as belonging to one of the classes.
NOISE, BORDER, CORE = 0, 1, 2

# Just a simple L2 norm.


def norm(p, q):
    return np.linalg.norm(np.subtract(p, q))


def density(eps, p, points):
    return sum(1 for q in points if norm(p, q) <= eps)


def label(points, eps, min_pts):
    """Returns an array lableing each point in the input as either NOISE, BORDER, CORE"""
    N = len(points)
    # Points are labeled noise by default.
    labels = np.zeros(N, dtype=np.int)

    # Label core points.
    for i, p in enumerate(points):
        if density(eps, p, points) >= min_pts:
            labels[i] = CORE

    # Label border points by marking each neighbour of a CORE point
    # as the 'best' of it's current label and BORDER.
    for i in np.arange(N)[labels == CORE]:
        for j in range(N):
            if norm(points[i], points[j]) <= eps:
                labels[j] = max(BORDER, labels[j])

    return labels


def _dbscan(points, eps, min_pts):
    N = len(points)
    labels = label(points, eps=eps, min_pts=min_pts)
    # The indices of CORE points.
    cores = np.arange(N)[labels == CORE]
    # Rather than adding edges as the algorithm in the book does,
    # I'll utilize a disjoint set to maintain information about the group each point belongs to.
    clusters = DisjointSet()

    # Assign cores in the vicinity of eachother to the same group.
    for a, i in enumerate(cores):
        for j in cores[a + 1:]:
            if norm(points[i], points[j]) <= eps:
                clusters.union(i, j)

    # For each border point, we'll simply assign it to the cluster
    # of the first CORE point we stumble upon in it's vicinity.
    for i in np.arange(N)[labels == BORDER]:
        for j in cores:
            if norm(points[i], points[j]) <= eps:
                clusters.union(i, j)
                break

    # Now, we have assigned every CORE and BORDER to a group. Now, we'll transform the disjoint set
    # into a list of lists, to make it easier for us to gauge what's in what.
    results = {}
    for i in np.arange(N)[labels != NOISE]:
        results.setdefault(clusters.find(i), []).append(i)

    # A tuple (groups, noise) where each of these is a list containing indices.
    return list(results.values()), np.arange(N)[labels == NOISE]


def dbscan_labels(points, eps, min_pts):
    display(
        pd.DataFrame(
            {
                'Point': f'$P_{{{i + 1}}}$',
                'Type': ['NOISE', 'BORDER', 'CORE'][l]
            } for i, l in enumerate(label(points, eps, min_pts))
        )
    )


def dbscan(points, eps, min_pts):
    groups, noise = _dbscan(points, eps, min_pts)
    print(f'Noise: {", ".join(f"P_{i + 1}" for i in noise)}')

    # We want to plot the various points with different symbols.
    for indices, marker in zip(groups, cycle('s^o1p*d')):
        x = [points[i][0] for i in indices]
        y = [points[i][1] for i in indices]
        plt.scatter(x, y, marker=marker)

    # Always draw the noise as 'X'-es
    x = [points[i][0] for i in noise]
    y = [points[i][1] for i in noise]
    plt.scatter(x, y, marker='x')

    plt.legend(list(map(str, range(len(groups)))) + ['NOISE'])

    # Just to get some nice formatting
    display(
        pd.DataFrame(
            {
                'Group': f'{{{", ".join(f"$P_{{{j + 1}}}$" for j in xs)}}}'
            } for xs in groups
        )
    )
    # Display the plot, duh.
    plt.show()
