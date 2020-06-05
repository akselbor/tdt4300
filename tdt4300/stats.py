from itertools import chain


def support_count(transactions, itemset):
    itemset = frozenset(itemset)
    return sum(1 for t in transactions if itemset.issubset(t))


def support(transactions, itemset):
    return support_count(transactions, itemset) / len(transactions)


def confidence(transactions, lhs, rhs):
    both = frozenset(chain(lhs, rhs))
    return support(transactions, both) / support(transactions, lhs)
