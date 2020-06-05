import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from itertools import combinations, count, product, cycle
from collections import OrderedDict
from disjoint_set import DisjointSet
from .uid import inc_id


def entropy(samples, as_str=False):
    _, counts = np.unique(samples[:, -1], return_counts=True)
    N = len(samples)
    p = counts / N
    if as_str:
        return f"- ({' + '.join(f'{c}/{N} * log2({c}/{N})' for c in counts)})"
    else:
        return -np.sum(p * np.log2(p))


def gini(samples, as_str=False):
    _, counts = np.unique(samples[:, -1], return_counts=True)
    N = len(samples)
    if as_str:
        return f"1 - {' - '.join(f'({i}/{N})^2' for i in counts)}"
    else:
        return 1 - sum((i / N)**2 for i in counts)


def name(attr, header):
    return header[attr] if header else str(attr)


def partition(samples, attr):
    """Partitions the samples based on their value of `attr`. Returns list of tuple (attr_value, samples)"""
    uniques, inverse = np.unique(samples[:, attr], return_inverse=True)

    return uniques, [samples[inverse == i] for i, _ in enumerate(uniques)]


def decision_tree(samples, header=None, show_calculations=False, dot=None, metric=entropy, dot_id=None):
    dot = dot or graphviz.Digraph()
    dot_id = dot_id or str(inc_id())

    uniques = np.unique(samples[:, -1])
    if len(uniques) == 1:
        dot.node(dot_id, label=f'{uniques[0]}', shape='none')
        return

    (_, width) = samples.shape
    base_metric = metric(samples)

    splits = [partition(samples, attr) for attr in range(width - 1)]
    entropies = [sum(len(subset) * metric(subset)
                     for subset in subsets) / len(samples) for _, subsets in splits]
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
        child_id = str(inc_id())
        dot.edge(dot_id, child_id, label=str(val))
        decision_tree(subset, header, show_calculations, dot, metric, child_id)

    return dot
