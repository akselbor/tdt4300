import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from itertools import combinations, count, product, cycle
from collections import OrderedDict
from disjoint_set import DisjointSet

counter = count()
uid = lambda: next(counter)

def flatten(xs):
    return [x for x, count in xs for _ in range(count)]

def fptree_dot(self, dot):
    """Constructs a Graphviz graph of an FPTree. Used to visualize FPTrees. However, since the implementation is
    mainly boilerplate and of no interest to the FP-Growth algorithm itself, it is left seperate here, rather than
    together with the actual FPTree implementation."""
    # Create a dot node for self
    dot.node(self.id(), f'{self.value}\n{self.frequency}')
        
    # Do so for each child, and connect the parent to the child.
    for child in self.children.values():
        fptree_dot(child, dot)
        #child.dot(dot)
        dot.edge(self.id(), child.id())
    
    # We make the root node responsible for drawing the edges between instances of each itemclass
    # As well as drawing the header table.
    if self.parent is None:
        subgraph = graphviz.Digraph()
        subgraph.attr(rank='same')
        subgraph.node(self.id(), f'{self.value}\n{self.frequency}')
        hid = str(uid())
        subgraph.edge(hid, self.id(), style='invis')
        # constraint='false' makes it such that graphviz doesn't take into account these    
        # edges when "shaping" the graph. It improves the graph layout for our purposes.
        dot.attr('edge', style='dotted', constraint='false')
        rows = ''.join(f'<TR><TD>{sum(v.frequency for v in vs)}</TD><TD PORT = "{k}_header">{k}</TD></TR>' for k, vs in self.elems.items())
        dot.node(hid, shape='none', label = f'<<TABLE><TR><TD>Count</TD><TD>Element</TD></TR>{rows}</TABLE>>')

    
        for k, v in self.elems.items():
            if v:
                dot.edge(f'{hid}:{k}_header', v[-1].id())
        
        
        for token, nodes in self.elems.items():
            rev = nodes[::-1]
            for (a, b) in zip(rev, rev[1:]):
                dot.edge(a.id(), b.id())
                
    
    return dot

sigma = lambda T, X: sum(1 for S in T if X.issubset(S))
s = lambda T, X: sigma(T, X) / len(T)
c = lambda T, X, Y: s(T, X.union(Y)) / s(T, X)

ident = lambda s: ''.join(sorted(list(s)))


class FPTree:
    """A trie with additional metadata to enable efficient creation of frequent itemsets."""
    # These enable the pretty graph drawings in Jupyter. The implementation of `fptree_dot` 
    # can be found at the top of the notebook, if it is of any interest.
    _repr_svg_ = lambda self: fptree_dot(self, graphviz.Digraph())._repr_svg_()
    def id(self):
        self._id = self._id if hasattr(self, '_id') else str(uid()) 
        return self._id
    
    @staticmethod
    def build_from(transactions, minsup = 0):
        """Builds an FPTree from a set of transactions, ensuring that item ordering is total."""
        counts = {x: sigma(transactions, {x}) for x in frozenset(y for ys in transactions for y in ys)}
        total_order = lambda x: (-counts[x], x)
        ordered_transactions = (
            sorted([x for x in xs if counts[x] >= minsup], key=total_order)
            for xs in sorted(transactions)
        )
        
        root = FPTree(elems = OrderedDict())
        for k in sorted(counts, key = total_order):
            if counts[k] >= minsup:
                root.elems.setdefault(k, [])
        
        for transaction in ordered_transactions:
            root.insert(transaction)
            
        return root
        
    def __init__(self, value=None, parent=None, frequency=0, elems=None):
        # A list for each type of token
        self.elems = OrderedDict() if elems is None else elems
        # The value at this node.
        self.value = value
        # The number of transactions "passing through" this node.
        self.frequency = frequency
        # The parent of this tree, if any. Only the root node has no parent.
        self.parent = parent
        # The children of this tree, each of whom is an instance of FPTree
        self.children = {}
    
    def insert(self, transaction):
        """Inserts a new item into the trie.
        NB: This does NOT take care to ensure that the item ordering in `self.elems` is consistent,
        which is necessary for the FP-Growth algorithm to operate correctly"""
        self.frequency += 1
        
        if not transaction:
            return
        
        token, *remainder = transaction
        
        if token in self.children:
            return self.children[token].insert(remainder)
        
        new = FPTree(token, self, 0, self.elems)
        self.elems.setdefault(token, []).append(new)
        self.children[token] = new
        new.insert(remainder)
    
    def expand(self, prefix = tuple()):
        """Expands the trie into a list of the transactions contained in it."""
        child_freq = sum(child.frequency for child in self.children.values())
        diff = self.frequency - child_freq
        
        if diff != 0 and prefix != tuple():
            yield (prefix, diff)
            
        for k, child in self.children.items():
            yield from child.expand(prefix + (k,))
            
    def branches(self, prefix = tuple()):
        """The various 'branches' that can be taken down this trie. Used to show the CFP in the step-through"""
        if self.children:
            for k, child in self.children.items():
                yield from child.branches(prefix + ((k, child.frequency),))
        elif prefix != tuple():
            yield prefix   
            
    def count(self, token):
        """Count the number of transactions `token` is included in within this trie."""
        return sum(node.frequency for node in self.elems.get(token, []))
        
    def ending_in(self, token):
        """Creates a copy of the subtree that contains all transactions truncated as ending in `token`,
        including `token`."""
        # This contains a mapping from a node in the original tree
        # to what is to be the corresponding node in the tree we're constructing.
        old_to_new = {}
        # This will be the `elems` for the copy
        elems = {}
        
        # The procedure is as follows:
        # For each leaf, we construct a copy of it's parent (recursively), effectively
        # computing a copy of the relevant part of the tree (path ending in `token`).
        # If no copy has been made, we'll need to construct one. However, if there has already
        # been made a copy of a node, we must make sure to alter that copy. This is achieved through `old_to_new`.
        # All the while we need to accurately track the frequency.
        # We defer filling `elems` until the end, as it is trivial to do once we've mapped out all the nodes.
        def build(node, freq):
            if node is None:
                return None
            
            # Construct copy of self and parent, and connect the two.
            parent = build(node.parent, freq)
            new = old_to_new.setdefault(node.id(), FPTree(node.value, parent, frequency=0, elems=elems))
            
            if parent:
                parent.children[new.value] = new
                
            new.frequency += freq
            return new
            
        # Build bottom-up from each leaf node.
        for node in self.elems[token]:
            build(node, node.frequency)
         
        # Give the copied subtree it's corresponding `elems` values
        for k, vs in self.elems.items():
            elems[k] = [old_to_new[v.id()] for v in vs if v.id() in old_to_new]
        
        return old_to_new[self.id()] 
    
    def cond(self, token, minsup=0):
        """A copy of the subtree that contains all that ends in `token` (truncated), *excluding* `token` itself."""
        # An FPTree of the prefix paths ending in `token`.
        prefix = self.ending_in(token)
        # Remove the `token` nodes (all of which are leaf nodes, 
        # and which constitute all the leaf nodes in `prefix`)
        for node in prefix.elems.pop(token, []):
            node.parent.children.pop(token, None)
            
        return FPTree.build_from(flatten(prefix.expand()), minsup)
            
    def _step_through(self, token, minsup, prefix = tuple()):
        prefix = prefix + (token,)
        # Conditional pattern base
        cbp = list(self.cond(token).expand())
        # Conditional FP-tree
        cfp = self.cond(token, minsup)
        branches = list(cfp.branches())
        # The table method used by the book takes a convoluted "shortcut" (not really)
        # for when the generated FP-tree is a single branch. Instead of simply recursing further,
        # they instead go like "meh, this is trivial we'll just compute it."
        if len(branches) == 1:
            yield (prefix, cbp, branches, [prefix + t for t in cfp.frequent_patterns(minsup)])
        else:
            yield (prefix, cbp, branches, [prefix + (e,) for e in cfp.elems])
            for tok in reversed(list(cfp.elems)):
                yield from cfp._step_through(tok, minsup, prefix)
                
    def step_through(self, minsup):
        for token in reversed(self.elems):
            if self.count(token) >= minsup:
                yield (None, None, None, [token])
                yield from self._step_through(token, minsup)
                   
    def frequent_patterns(self, minsup, prefix = tuple()):
        for token in reversed(self.elems):
            if self.count(token) >= minsup:
                new = prefix + (token,)
                yield new
                yield from self.cond(token).frequent_patterns(minsup, prefix=new)

def latex(xs):
    return f'${xs}$'

class PrettyFPTree:
    _repr_svg_ = lambda self: self.root._repr_svg_()
    
    def __init__(self, transactions, minsup):
        self.root = FPTree.build_from(transactions, minsup)
        
    def frequent_patterns(self, minsup):
        root = self.root
        transactions = flatten(root.expand())
        return pd.DataFrame(
            {
                'len': len(ident(pat)), 
                'Frequent Set': latex(','.join(sorted(pat))), 
                'Support Count': sigma(transactions, frozenset(pat))
            } for pat in root.frequent_patterns(minsup)
        ).sort_values(['len', 'Frequent Set']).drop(columns=['len']).reset_index(drop=True)
    
    def step_through(self, minsup):
        root = self.root
        transactions = flatten(root.expand())
        return pd.DataFrame(
            {
                'Item': '-' if not i else latex(', '.join(str(item) for item in i)), 
                'Conditional Base Pattern': '-' if not cbp else latex(', '.join('\\{' + ', '.join(pat) + f': {count}' + '\\}' for (pat, count) in cbp)),
                'Conditional FP-Tree': '-' if not cfp else latex(', '.join(' \\langle ' + ', '.join(f'{pat}: {c}' for pat, c in branch) + ' \\rangle ' for branch in cfp)),
                'Frequent Patterns Generated': '-' if not pats else latex(', '.join('\\{' + ', '.join(pat) + f': {sigma(transactions, frozenset(pat))}'+ '\\}' for pat in pats)),
            } for (i, cbp, cfp, pats) in root.step_through(minsup)
        )

def fp_growth(transactions, minsup, display_tree = True, display_steps = True, display_fp = False):
    root = PrettyFPTree(transactions, minsup)
    if display_tree:
        display(root)
    if display_steps:
        display(root.step_through(minsup))
    if display_fp:
        display(root.frequent_patterns(minsup))


def apriori_fkfk(transactions, minsup, prev = [], dot = None):
    """This is horrible"""
    dot = dot or graphviz.Digraph(graph_attr={'ordering': 'out'})
    ident = lambda items: ' '.join(items)
    # Prev is a list of frozensets
    if not prev:
        layer = sorted(frozenset([x for xs in transactions for x in xs]))
        root = str(uid())
        dot.node(root, label='<<I>{}</I>>')
        for node in layer:
            dot.edge(root, node)
            sup = sigma(transactions, frozenset({node}))
            if sup >= minsup:
                dot.node(node, label=f'{node}\n{sup}')
            else:
                dot.node(node, label=f'<<S>{node}</S><BR/>{sup}>', shape='none')
                
        apriori_fkfk(transactions, minsup, [[item] for item in layer if sigma(transactions, frozenset({item})) >= minsup], dot)
        return dot
    
    current = []
    for (a, b) in combinations(prev, r=2):
        if a[:-1] != b[:-1]:
            continue
            
        new = sorted(a + b[-1:])
        identifier = ident(new)
        sup = sigma(transactions, frozenset(new))
        
        if sup >= minsup:
            current.append(new)
            dot.node(identifier, label=f'{identifier}\n{sup}')
        else:
            dot.node(identifier, label=f'<<S>{identifier}</S><BR/>{sup}>', shape='none')
        
        dot.edge(ident(a), identifier)
        dot.edge(ident(b), identifier)
            
    if current:
        apriori_fkfk(transactions, minsup, current, dot)
    return dot

def entropy(samples, as_str = False):
    uniques, counts = np.unique(samples[:, -1], return_counts=True)
    N = len(samples)
    p = counts / N
    if as_str:
        return f"- ({' + '.join(f'{c}/{N} * log2({c}/{N})' for c in counts)})"
    else:
        return -np.sum(p * np.log2(p))
    
def gini(samples, as_str = False):
    uniques, counts = np.unique(samples[:, -1], return_counts=True)
    N = len(samples)
    if as_str:
        return f"1 - {' - '.join(f'({i}/{N})^2' for i in counts)}"
    else:
        return 1 - sum((i / N)**2 for i in counts)
    
def name(attr, header):
    return header[attr] if header else str(attr)

def partition(samples, attr):
    """Partitions the samples based on their value of `attr`. Returns list of tuple (attr_value, samples)"""
    uniques, inverse = np.unique(samples[:, attr], return_inverse = True)
    
    return uniques, [samples[inverse == i] for i, _ in enumerate(uniques)]

def decision_tree(samples, header = None, show_calculations = False, dot = None, metric = entropy, dot_id = None):
    dot = dot or graphviz.Digraph()
    dot_id = dot_id or str(uid())
    
    uniques = np.unique(samples[:, -1])
    if len(uniques) == 1:
        dot.node(dot_id, label=f'{uniques[0]}', shape='none')
        return
    
    (_, width) = samples.shape
    base_metric = metric(samples)
    
    splits = [partition(samples, attr) for attr in range(width - 1)]
    entropies = [sum(len(subset) * metric(subset) for subset in subsets) / len(samples) for _, subsets in splits]
    i = max(range(len(splits)), key=lambda i: base_metric - entropies[i])
    (vals, subsets) = splits[i]
    
    # First we'll create the dot node for the current node.
    if show_calculations:
        def calcs(split):
            (_, subsets) = splits[split]
            return ' + '.join(f'{len(subset)} * ({metric(subset, as_str=True)})' for subset in subsets)
        # Absolutely horrendous but it works for my purposes here so idgaf.
        metric_name = metric.__name__.title()
        label = f"""<
            <TABLE>
                <TR><TD COLSPAN="2"><B>{name(i, header)}</B>   is choosen as split attribute, since is has the least Post Split {metric_name}:</TD></TR>
                <TR><TD> <B>Split Attribute</B>   </TD><TD><B>Post Split {metric_name} (Gain = Base {metric_name} - Post Split {metric_name})</B></TD></TR>
                {''.join(f'<TR><TD>{name(i, header)}</TD><TD>{entropies[i]:.4f} = ({calcs(i)}) / {len(samples)}</TD></TR>' for (i, _) in enumerate(splits))}
            </TABLE>
        >"""
    else:
        label = f'{name(i, header)}'
    
    dot.node(dot_id, label=label, shape='none' if show_calculations else 'oval')
    
    for val, subset in zip(vals, subsets):
        child_id = str(uid())
        dot.edge(dot_id, child_id, label=str(val))
        decision_tree(subset, header, show_calculations, dot, metric, child_id)

    return dot

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

def kmeans(data, centroids, iterations = -1):
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
    def __init__(self, items, subclusters = [], name = ''):
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
        self._name = self._name or ''.join(sorted(child.name() for child in self.subclusters))
        return self._name
    
    def dot(self, dot = None):
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
    
def hac(table, metric, print_progress = False):
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

# Covariance and standard deviation.
cov = lambda x, y: 1 / (len(x) - 1) * sum(np.subtract(x, np.mean(x)) * np.subtract(y, np.mean(y)))
stddev = lambda x: np.sqrt(cov(x, x))

# Similarity measures
corr = lambda x, y: cov(x, y) / (stddev(x) * stddev(y))
euclidean = lambda x, y: np.linalg.norm(np.subtract(x, y))
cosine = lambda x, y: np.array(x).dot(np.array(y)) / (np.linalg.norm(x) * np.linalg.norm(y))
jaccard = lambda x, y: sum(np.logical_and(np.equal(x, 1), np.equal(y, 1)) / (sum(np.logical_or(np.not_equal(x, 0), np.not_equal(y, 0)))))