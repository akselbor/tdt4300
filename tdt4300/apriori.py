import graphviz
from itertools import combinations
from .stats import support_count


def apriori(transactions, minsup, candidate_generation='fkfk'):
    """Constructs a graphviz graph showing the apriori algorithm using a particular candidate generation method."""
    if candidate_generation != 'fkfk':
        raise ValueError(
            f"Unknown candidate generation '{candidate_generation}'")

    one_itemsets = sorted(frozenset(x for xs in transactions for x in xs))
    dot = graphviz.Digraph(graph_attr={'ordering': 'out'})
    dot.node('ROOT', label='<<I>{}</I>>')
    for node in one_itemsets:
        dot.edge('ROOT', node)
        sup = support_count(transactions, {node})
        # We will use strike-through on the label of pruned itemsets, and no shape (node border).
        label = f'{node}\n{sup}' if sup >= minsup else f'<<S>{node}</S><BR/>{sup}>'
        shape = 'oval' if sup >= minsup else 'none'
        dot.node(node, label=label, shape=shape)

    return apriori_fkfk(transactions, minsup, [[item] for item in one_itemsets if support_count(
        transactions, {item}) >= minsup], dot)


def apriori_fkfk(transactions, minsup, prev, dot):
    """This is horrible"""
    current = []
    for (a, b) in combinations(prev, r=2):
        if a[:-1] != b[:-1]:
            continue

        new = sorted(a + b[-1:])
        identifier = ' '.join(new)
        sup = support_count(transactions, frozenset(new))

        if sup >= minsup:
            current.append(new)
            dot.node(identifier, label=f'{identifier}\n{sup}')
        else:
            dot.node(
                identifier, label=f'<<S>{identifier}</S><BR/>{sup}>', shape='none')

        dot.edge(' '.join(a), identifier)
        dot.edge(' '.join(b), identifier)

    if current:
        apriori_fkfk(transactions, minsup, current, dot)

    return dot
