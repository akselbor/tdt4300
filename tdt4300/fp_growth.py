import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from itertools import combinations, count, product, cycle
from collections import OrderedDict
from disjoint_set import DisjointSet
from .stats import support, support_count, confidence
from functools import wraps


class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        return self.gen.__iter__()


def inject_bound_method_to_generator(method_name, val):
    def decorate(function):
        @wraps(function)
        def inner(*args, **kwargs):
            wrapper = GeneratorWrapper(function(*args, **kwargs))
            setattr(wrapper, method_name, val.__get__(wrapper))
            return wrapper
        return inner
    return decorate


def inject_repr_html(val):
    def decorate(function):
        @wraps(function)
        def inner(self, minsup, *args, **kwargs):
            wrapper = GeneratorWrapper(function(self, minsup, *args, **kwargs))
            setattr(wrapper, '_repr_html_',
                    val.__get__((wrapper, self, minsup)))
            return wrapper
        return inner
    return decorate


def str_id(x):
    return str(id(x))


def flatten(xs):
    return [x for x, count in xs for _ in range(count)]


def inject(attr, val, fn):
    f = fn
    setattr(f, attr, val)
    setattr
    return f


def formatted_frequent_patterns(args):
    (gen, self, _) = args
    transactions = flatten(self.expand())
    return pd.DataFrame(
        {
            'len': len(''.join(sorted(list(pat)))),
            'Frequent Set': latex(','.join(sorted(pat))),
            'Support Count': support_count(transactions, frozenset(pat))
        } for pat in gen
    ).sort_values(['len', 'Frequent Set']).drop(columns=['len']).reset_index(drop=True)._repr_html_()


def formatted_step_through(args):
    (gen, self, _) = args
    transactions = flatten(self.expand())
    return pd.DataFrame(
        {
            'Item': '-' if not i else latex(', '.join(str(item) for item in i)),
            'Conditional Base Pattern': '-' if not cbp else latex(', '.join('\\{' + ', '.join(pat) + f': {count}' + '\\}' for (pat, count) in cbp)),
            'Conditional FP-Tree': '-' if not cfp else latex(', '.join(' \\langle ' + ', '.join(f'{pat}: {c}' for pat, c in branch) + ' \\rangle ' for branch in cfp)),
            'Frequent Patterns Generated': '-' if not pats else latex(', '.join('\\{' + ', '.join(pat) + f': {support_count(transactions, frozenset(pat))}' + '\\}' for pat in pats)),
        } for (i, cbp, cfp, pats) in gen
    )._repr_html_()


class FPTree:
    """A trie with additional metadata to enable efficient creation of frequent itemsets."""
    # These enable the pretty graph drawings in Jupyter. The implementation of `fptree_dot`
    # can be found at the top of the notebook, if it is of any interest.

    def _repr_dot_(self, dot):
        """Constructs a Graphviz graph of an FPTree. Used to visualize FPTrees. However, since the implementation is
        mainly boilerplate and of no interest to the FP-Growth algorithm itself, it is left seperate here, rather than
        together with the actual FPTree implementation."""
        # Create a dot node for self
        dot.node(str_id(self), f'{self.value}\n{self.frequency}')

        # Do so for each child, and connect the parent to the child.
        for child in self.children.values():
            child._repr_dot_(dot)
            dot.edge(str_id(self), str_id(child))

        # We make the root node responsible for drawing the edges between instances of each itemclass
        # As well as drawing the header table.
        if self.parent is None:
            subgraph = graphviz.Digraph()
            subgraph.attr(rank='same')
            subgraph.node(str_id(self), f'{self.value}\n{self.frequency}')
            hid = 'HEADER'
            subgraph.edge(hid, str_id(self), style='invis')
            # constraint='false' makes it such that graphviz doesn't take into account these
            # edges when "shaping" the graph. It improves the graph layout for our purposes.
            dot.attr('edge', style='dotted', constraint='false')
            rows = ''.join(
                f'<TR><TD>{sum(v.frequency for v in vs)}</TD><TD PORT = "{k}_header">{k}</TD></TR>' for k, vs in self.elems.items())
            dot.node(hid, shape='none',
                     label=f'<<TABLE><TR><TD>Count</TD><TD>Element</TD></TR>{rows}</TABLE>>')

            for k, v in self.elems.items():
                if v:
                    dot.edge(f'{hid}:{k}_header', str_id(v[-1]))

            for token, nodes in self.elems.items():
                rev = nodes[::-1]
                for (a, b) in zip(rev, rev[1:]):
                    dot.edge(str_id(a), str_id(b))

        return dot

    def _repr_svg_(self): return self._repr_dot_(
        graphviz.Digraph())._repr_svg_()

    @staticmethod
    def build_from(transactions, minsup=0):
        """Builds an FPTree from a set of transactions, ensuring that item ordering is total."""
        counts = {x: support_count(transactions, {x})
                  for x in frozenset(y for ys in transactions for y in ys)}

        def total_order(x): return (-counts[x], x)
        ordered_transactions = (
            sorted([x for x in xs if counts[x] >= minsup], key=total_order)
            for xs in sorted(transactions)
        )

        root = FPTree(elems=OrderedDict())
        for k in sorted(counts, key=total_order):
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

    def expand(self, prefix=tuple()):
        """Expands the trie into a list of the transactions contained in it."""
        child_freq = sum(child.frequency for child in self.children.values())
        diff = self.frequency - child_freq

        if diff != 0 and prefix != tuple():
            yield (prefix, diff)

        for k, child in self.children.items():
            yield from child.expand(prefix + (k,))

    def branches(self, prefix=tuple()):
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
            new = old_to_new.setdefault(str_id(node), FPTree(
                node.value, parent, frequency=0, elems=elems))

            if parent:
                parent.children[new.value] = new

            new.frequency += freq
            return new

        # Build bottom-up from each leaf node.
        for node in self.elems[token]:
            build(node, node.frequency)

        # Give the copied subtree it's corresponding `elems` values
        for k, vs in self.elems.items():
            elems[k] = [old_to_new[str_id(v)]
                        for v in vs if str_id(v) in old_to_new]

        return old_to_new[str_id(self)]

    def cond(self, token, minsup=0):
        """A copy of the subtree that contains all that ends in `token` (truncated), *excluding* `token` itself."""
        # An FPTree of the prefix paths ending in `token`.
        prefix = self.ending_in(token)
        # Remove the `token` nodes (all of which are leaf nodes,
        # and which constitute all the leaf nodes in `prefix`)
        for node in prefix.elems.pop(token, []):
            node.parent.children.pop(token, None)

        return FPTree.build_from(flatten(prefix.expand()), minsup)

    def _step_through(self, token, minsup, prefix=tuple()):
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

    @inject_repr_html(formatted_step_through)
    def step_through(self, minsup):
        for token in reversed(self.elems):
            if self.count(token) >= minsup:
                yield (None, None, None, [token])
                yield from self._step_through(token, minsup)

    @inject_repr_html(formatted_frequent_patterns)
    def frequent_patterns(self, minsup, prefix=tuple()):
        for token in reversed(self.elems):
            if self.count(token) >= minsup:
                new = prefix + (token,)
                yield new
                yield from self.cond(token).frequent_patterns(minsup, prefix=new)


def latex(xs):
    return f'${xs}$'


class PrettyFPTree:
    def _repr_svg_(self): return self.root._repr_svg_()

    def __init__(self, transactions, minsup):
        self.root = FPTree.build_from(transactions, minsup)

    def frequent_patterns(self, minsup):
        root = self.root
        transactions = flatten(root.expand())
        return pd.DataFrame(
            {
                'len': len(''.join(sorted(list(pat)))),
                'Frequent Set': latex(','.join(sorted(pat))),
                'Support Count': support_count(transactions, frozenset(pat))
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
                'Frequent Patterns Generated': '-' if not pats else latex(', '.join('\\{' + ', '.join(pat) + f': {support_count(transactions, frozenset(pat))}' + '\\}' for pat in pats)),
            } for (i, cbp, cfp, pats) in root.step_through(minsup)
        )


def fp_growth(transactions, minsup, display_tree=True, display_steps=True, display_fp=False):
    root = PrettyFPTree(transactions, minsup)
    if display_tree:
        display(root)
    if display_steps:
        display(root.step_through(minsup))
    if display_fp:
        display(root.frequent_patterns(minsup))
