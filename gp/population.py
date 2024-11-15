#!/usr/bin/env python3

# Filename: population.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import numpy as np

from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score

from .gene import Gene, Node
from .parameters import TERMINAL, NON_TERMINAL, TREE_MAX_DEPTH, TOURNAMENT_SIZE, TERMINAL_PROB

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

    return np.random.choice(nodes)


def grow(depth) -> Node:
    """
    Grow strategy for the genetic programming algorithm

    @param depth: The depth of the tree
    @return: A random node
    """
    if depth == 0:
        return Node(np.random.choice(TERMINAL), depth=TREE_MAX_DEPTH)

    if np.random.random() < TERMINAL_PROB:
        return Node(np.random.choice(TERMINAL), depth=TREE_MAX_DEPTH - depth)

    left = grow(depth - 1)
    right = grow(depth - 1)

    return Node(
        np.random.choice(NON_TERMINAL),
        depth=TREE_MAX_DEPTH - depth,
        left=left,
        right=right,
    )


def full(depth) -> Node:
    """
    Full strategy for the genetic programming algorithm

    @param depth: The depth of the tree
    @return: A random node
    """
    if depth == 0:
        return Node(np.random.choice(TERMINAL), depth=TREE_MAX_DEPTH)

    left = full(depth - 1)
    right = full(depth - 1)

    return Node(
        np.random.choice(NON_TERMINAL),
        depth=TREE_MAX_DEPTH - depth,
        left=left,
        right=right,
    )


def half_and_half(depth) -> Node:
    """
    Half-and-half strategy for the genetic programming algorithm

    @param depth: The depth of the tree
    @return: A random node
    """
    return grow(depth) if np.random.random() < 0.5 else full(depth)


def generate_random_tree(depth, strategy=grow) -> Node:
    """
    Generate a random tree with the given depth

    @param depth: The depth of the tree
    @param strategy: The strategy to generate the tree
    @return: A random tree
    """
    return strategy(depth)


def generate_initial_population(population_size, strategy) -> List[Gene]:
    """
    Generate the initial population of the genetic programming algorithm

    @param population_size: The size of the population
    @return: A list with the initial population
    """

    if strategy == "grow":
        return [
            Gene(generate_random_tree(TREE_MAX_DEPTH, grow))
            for _ in range(population_size)
        ]

    elif strategy == "full":
        return [
            Gene(generate_random_tree(TREE_MAX_DEPTH, full))
            for _ in range(population_size)
        ]

    elif strategy == "half_and_half":
        genes_per_depth = population_size // TREE_MAX_DEPTH
        remainder = population_size % TREE_MAX_DEPTH
        population = []

        # Generate genes at each depth alternating between the grow and full strategies
        for depth in range(1, TREE_MAX_DEPTH + 1):
            grow_genes = genes_per_depth // 2
            full_genes = genes_per_depth - grow_genes

            # Add the remainder to the last depth to ensure the correct population size
            if depth == TREE_MAX_DEPTH:
                full_genes += remainder

            population += [
                Gene(generate_random_tree(depth, grow)) for _ in range(grow_genes)
            ] + [Gene(generate_random_tree(depth, full)) for _ in range(full_genes)]

        # Ensure that the population has the correct size
        if len(population) > population_size:
            population = population[:population_size]

        return population

    else:
        raise ValueError("Invalid strategy")


def selection_tournament(
    population,
    tournament_size=TOURNAMENT_SIZE,
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
