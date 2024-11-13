#!/usr/bin/env python3

# Filename: operators.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import random

from copy import deepcopy

from .gene import *
from .parameters import *
from .population import *


def get_leaf_nodes(node, nodes):
    """
    Get all leaf nodes in the tree

    @param node: The current node
    @param nodes: The list of leaf nodes
    """
    if node.is_leaf():
        nodes.append(node)
        return

    if node.left:
        get_leaf_nodes(node.left, nodes)

    if node.right:
        get_leaf_nodes(node.right, nodes)


def bfs(node, depth, max_depth, nodes):
    """
    Breadth-first search in the tree

    @param node: The current node
    @param depth: The current depth
    @param max_depth: The maximum depth
    @param nodes: The list of nodes with the given depth
    """
    if depth == max_depth:
        nodes.append(node)
        return

    if node.left:
        bfs(node.left, depth + 1, max_depth, nodes)

    if node.right:
        bfs(node.right, depth + 1, max_depth, nodes)


def select_random_subtree(gene, depth=None) -> Node:
    """
    Select a random subtree from the given node

    @param node: The node to select the subtree
    @param depth: The depth of the nodes that will be selected.
                  If None, all nodes are eligible
                  Note: Used to prevent broating
    """
    nodes = []
    nodes.append(gene.root_node)

    if depth is None:
        depth = gene.root_node.get_depth()

    bfs(gene.root_node, 0, depth, nodes)

    return random.choice(nodes)


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

    child_base.recalculate_height()

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
    gene.recalculate_height()
