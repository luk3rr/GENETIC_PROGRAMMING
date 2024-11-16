#!/usr/bin/env python3

# Filename: population.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import numpy as np

from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score

from .gene import Gene, Node
from .parameters import (
    TERMINAL,
    NON_TERMINAL,
    TREE_MAX_DEPTH,
    TREE_MIN_DEPTH,
    TERMINAL_PROB,
)


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
    assert depth is None or (depth <= TREE_MAX_DEPTH), (
        "Invalid depth value. "
        "The depth must be less than or equal to the maximum depth of the tree"
    )

    nodes = []

    if depth is None:
        depth = gene.root_node.depth

    bfs(gene.root_node, 0, depth, nodes)

    return np.random.choice(nodes)


def grow(min_depth, max_depth) -> Node:
    """
    Grow strategy for the genetic programming algorithm

    @param depth: The min depth of the tree
    @param depth: The max depth of the tree
    @return: A random node with depth between min_depth and max_depth
    """
    return _grow(0, min_depth, max_depth)


def _grow(current_depth, min_depth, max_depth) -> Node:
    """
    Grow strategy for the genetic programming algorithm

    @param depth: The current depth of the tree
    @param min_depth: The min depth of the tree
    @param max_depth: The max depth of the tree
    @return: A random node with depth between min_depth and max_depth
    """
    if current_depth == max_depth:
        return Node(np.random.choice(TERMINAL), depth=current_depth)

    # The terminals have a probability of being chosen only
    # if the minimum depth is reached
    if current_depth >= min_depth and np.random.random() < TERMINAL_PROB:
        return Node(np.random.choice(TERMINAL), depth=current_depth)

    left = _grow(current_depth + 1, min_depth, max_depth)
    right = _grow(current_depth + 1, min_depth, max_depth)

    return Node(
        np.random.choice(NON_TERMINAL),
        depth=current_depth,
        left=left,
        right=right,
    )


def full(depth) -> Node:
    """
    Full strategy for the genetic programming algorithm

    @param depth: The depth of the tree
    @return: A random node with depth equal to the given depth
    """
    if depth > TREE_MAX_DEPTH:
        raise ValueError("Invalid depth")

    return _full(0, depth)


def _full(current_depth, max_depth) -> Node:
    """
    Full strategy for the genetic programming algorithm

    @param depth: The current depth of the tree
    @return: A random node
    """
    if current_depth == max_depth:
        return Node(np.random.choice(TERMINAL), depth=max_depth)

    left = _full(current_depth + 1, max_depth)
    right = _full(current_depth + 1, max_depth)

    return Node(
        np.random.choice(NON_TERMINAL),
        depth=current_depth,
        left=left,
        right=right,
    )


def half_and_half(depth, min_depth, max_depth) -> Node:
    """
    Half-and-half strategy for the genetic programming algorithm

    @param depth: The depth of the tree if full strategy is selected
    @param min_depth: The min depth of the tree if grow strategy is selected
    @param max_depth: The max depth of the tree if grow strategy is selected
    @return: A random node
    """
    return grow(min_depth, max_depth) if np.random.random() < 0.5 else full(depth)


def generate_random_tree(
    strategy, depth, min_depth=TREE_MIN_DEPTH, max_depth=TREE_MAX_DEPTH
) -> Node:
    """
    Generate a random tree with the given depth

    @param strategy: The strategy to generate the tree
    @param depth: The depth of the tree, if the strategy is full
    @param min_depth: The min depth of the tree, if the strategy is grow
    @param max_depth: The max depth of the tree, if the strategy is grow
    @return: A random tree
    """
    if strategy == grow:
        return grow(min_depth, max_depth)

    elif strategy == full:
        return full(depth)

    elif strategy == half_and_half:
        return half_and_half(depth, min_depth, max_depth)

    else:
        raise ValueError("Invalid strategy")


def generate_initial_population(population_size, strategy) -> List[Gene]:
    """
    Generate the initial population of the genetic programming algorithm

    @param population_size: The size of the population
    @return: A list with the initial population
    """
    if strategy == "grow":
        return [
            Gene(generate_random_tree(grow, TREE_MAX_DEPTH))
            for _ in range(population_size)
        ]

    elif strategy == "full":
        return [
            Gene(generate_random_tree(full, TREE_MAX_DEPTH))
            for _ in range(population_size)
        ]

    elif strategy == "half_and_half":
        depths = (
            (TREE_MAX_DEPTH - TREE_MIN_DEPTH) if TREE_MIN_DEPTH < TREE_MAX_DEPTH else 1
        )
        genes_per_depth = population_size // depths
        remaining = population_size % depths
        population = []

        # Generate genes at each depth with the half-and-half strategy
        for depth in range(TREE_MIN_DEPTH, TREE_MAX_DEPTH):
            population += [
                Gene(generate_random_tree(half_and_half, depth))
                for _ in range(genes_per_depth)
            ]

        # Generate the remaining genes
        population += [
            Gene(generate_random_tree(half_and_half, TREE_MAX_DEPTH))
            for _ in range(remaining)
        ]

        return population

    else:
        raise ValueError("Invalid strategy")


def selection_tournament(
    population,
    tournament_size,
    problem_type="max",
) -> Gene:
    """
    Perform the tournament selection and return a list of selected genes

    @param population: The population to select from
    @param tournament_size: The size of the tournament
    @param problem_type: The type of problem, minimization or maximization ("min" or "max")
    @return: The best gene selected
    NOTE: Ensure that the fitnesses of the genes are updated before calling this function
    """
    # Random selection of the subset of genes for the tournament
    tournament = np.random.choice(population, tournament_size, replace=False)

    if problem_type == "min":
        return min(tournament, key=lambda gene: gene.fitness)
    elif problem_type == "max":
        return max(tournament, key=lambda gene: gene.fitness)
    else:
        raise ValueError("Invalid problem type")


def count_duplicated_genes(population) -> Tuple[int, List[int]]:
    """
    Count the number of duplicated genes in the population

    @param population: The population to count the duplicated genes
    @return: The number of duplicated genes and the indexes of the duplicated genes
    """
    unique_genes = set()
    duplicated_genes = 0
    duplicated_indexes = []

    for gene in population:
        if gene in unique_genes:
            duplicated_indexes.append(population.index(gene))
            duplicated_genes += 1
        else:
            unique_genes.add(gene)

    return duplicated_genes, duplicated_indexes


def evaluate_fitness(gene, data, true_labels):
    """
    Evaluate the fitness of the gene

    @param gene: The gene to evaluate
    @param data: The data to evaluate
    @param true_labels: The true labels of the data
    @return: The fitness of the gene
    """
    distance_matrix = gene.get_distance_matrix(data)
    clustering = AgglomerativeClustering(
        n_clusters=len(set(true_labels)), metric="precomputed", linkage="average"
    )

    return float(v_measure_score(true_labels, clustering.fit_predict(distance_matrix)))
