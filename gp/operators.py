#!/usr/bin/env python3

# Filename: operators.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import random

from copy import deepcopy

from .gene import *
from .parameters import *
from .population import *
from .utils import *

def crossover(parent1, parent2) -> Gene:
    """
    Crossover between two parents

    @param parent1: The first parent
    @param parent2: The second parent
    @return: The child gene
    """

    # Choose the max depth to prevent broating
    height = min(parent1.height, parent2.height)

    # 0.5 of chance to choose parent 1 or parent 2 as the base for the child
    # Then choose a random node from the base and a random node from the other parent
    # and swap the nodes
    if random.random() < 0.5:
        child_base = deepcopy(parent1)
        subtree_parent = select_random_subtree(parent2, height)
    else:
        child_base = deepcopy(parent2)
        subtree_parent = select_random_subtree(parent1, height)

    subtree_child = select_random_subtree(child_base, height)

    subtree_child.value = subtree_parent.value
    subtree_child.left = subtree_parent.left
    subtree_child.right = subtree_parent.right

    child_base.calculate_tree_height()

    return child_base


def one_point_mutation(gene):
    """
    Perform a one-point mutation in the gene

    @param gene: The gene to mutate
    """
    subtree = select_random_subtree(gene)

    if subtree.is_leaf():
        subtree.value = random.choice(TERMINAL)
    else:
        subtree.value = random.choice(NON_TERMINAL)


def expand_mutation(gene):
    """
    Perform an expand mutation in the gene

    @param gene: The gene to mutate
    """
    subtree = select_random_subtree(gene)

    # Get all leaf nodes in the subtree and select a random one
    nodes = []

    get_leaf_nodes(subtree, nodes)

    leaf = random.choice(nodes)

    new_random = generate_random_tree(
        random.randint(1, TREE_MAX_DEPTH - gene.root_node.get_depth())
        if TREE_MAX_DEPTH > gene.root_node.get_depth()
        else 1
    )

    leaf.value = new_random.value
    leaf.left = new_random.left
    leaf.right = new_random.right


def shrink_mutation(gene):
    """
    Perform a shrink mutation in the gene

    @param gene: The gene to mutate
    """
    subtree = select_random_subtree(gene)

    subtree.left = None
    subtree.right = None
    subtree.value = random.choice(TERMINAL)


def mutate(gene, strategy=one_point_mutation):
    """
    Mutate a gene

    @param gene: The gene to mutate
    @param strategy: The mutation strategy
    """
    strategy(gene)
    gene.calculate_tree_height()

def mutate_random_strategy(gene):
    """
    Mutate a gene with a random mutation strategy

    @param gene: The gene to mutate
    """
    strategy = random.choice([one_point_mutation, expand_mutation, shrink_mutation])
    mutate(gene, strategy)
