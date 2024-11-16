#!/usr/bin/env python3

# Filename: operators.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from copy import deepcopy
from typing import Tuple

import numpy as np

from gp.utils import update_nodes_depth

from .gene import Gene
from .population import (
    grow,
    select_random_subtree,
    get_leaf_nodes,
    generate_random_tree,
    selection_tournament,
    evaluate_fitness,
)
from .parameters import NON_TERMINAL, TREE_MAX_DEPTH


def crossover(parent1, parent2) -> Gene:
    """
    Crossover between two parents

    @param parent1: The first parent
    @param parent2: The second parent
    @return: The child gene
    """
    # Choose the max depth to prevent broating
    max_depth = min(parent1.height, parent2.height)

    # 0.5 of chance to choose parent 1 or parent 2 as the base for the child
    # Then choose a random node from the base and a random node from the other parent
    # and swap the nodes
    if np.random.random() < 0.5:
        child_base = deepcopy(parent1)
        donor_parent = deepcopy(parent2)
    else:
        child_base = deepcopy(parent2)
        donor_parent = deepcopy(parent1)

    subtree_child = select_random_subtree(child_base, max_depth)
    subtree_donor = select_random_subtree(donor_parent, max_depth)

    subtree_child.value = subtree_donor.value
    subtree_child.left = subtree_donor.left
    subtree_child.right = subtree_donor.right

    child_base.calculate_tree_height()
    update_nodes_depth(child_base.root_node)

    assert (
        child_base.height <= TREE_MAX_DEPTH
    ), "Child height is greater than the max depth."

    return child_base


def one_point_mutation(gene, terminals):
    """
    Perform a one-point mutation in the gene

    @param gene: The gene to mutate
    """
    subtree = select_random_subtree(gene)

    if subtree.is_leaf():
        subtree.value = np.random.choice(terminals)
    else:
        subtree.value = np.random.choice(NON_TERMINAL)

    gene.calculate_tree_height()

    assert (
        gene.height <= TREE_MAX_DEPTH
    ), "Gene height is greater than the max depth."


def expand_mutation(gene, terminals):
    """
    Perform an expand mutation in the gene

    @param gene: The gene to mutate
    """
    # Get all leaf nodes in the subtree and select a random one
    nodes = []

    get_leaf_nodes(gene.root_node, nodes)

    leaf = np.random.choice(nodes)

    max_allowed_depth = TREE_MAX_DEPTH - leaf.depth

    depth = (
        np.random.randint(1, max_allowed_depth + 1)
        if max_allowed_depth > 0
        else 0
    )

    # Generate a random tree with half and half method and the depth value
    new_random = generate_random_tree(grow, terminals, depth, depth, depth)

    leaf.value = new_random.value
    leaf.left = new_random.left
    leaf.right = new_random.right

    gene.calculate_tree_height()

    assert (
        gene.height <= TREE_MAX_DEPTH
    ), f"Gene height is greater than the max depth:"


def shrink_mutation(gene, terminals):
    """
    Perform a shrink mutation in the gene

    @param gene: The gene to mutate
    """
    subtree = select_random_subtree(gene)

    subtree.left = None
    subtree.right = None
    subtree.value = np.random.choice(terminals)

    gene.calculate_tree_height()


def mutate(gene, terminals, strategy=one_point_mutation):
    """
    Mutate a gene

    @param gene: The gene to mutate
    @param strategy: The mutation strategy
    """
    strategy(gene, terminals)


def mutate_random_strategy(gene, terminals):
    """
    Mutate a gene with a random mutation strategy

    @param gene: The gene to mutate
    """
    strategy = np.random.choice(
        np.array([one_point_mutation, expand_mutation, shrink_mutation])
    )
    mutate(gene, terminals, strategy)


def generate_child(
        population, terminals, data, true_labels, tournament_size, crossover_prob, mutation_prob, seed
) -> Tuple[Gene, Tuple[Gene, Gene]] | None:
    """
    Generates a new child by selecting two parents, applying crossover and mutation, and evaluating fitness.

    @param population: The current population of genes
    @param data: The data to use for fitness evaluation
    @param true_labels: The true labels to use for fitness evaluation
    @param crossover_prob: The probability of applying crossover
    @param mutation_prob: The probability of applying mutation
    @param seed: The seed to use for random number generation
    NOTE: If this function is parallelized with ThreadPoolExecutor, the seed defined
          here will override the global seed. Therefore, parallelize with ProcessPoolExecutor.
    @return: A tuple in the form (child, (parent1, parent2)) if crossover occurs, None otherwise
    """
    np.random.seed(seed)

    # Select two parents
    parent1 = selection_tournament(population, tournament_size)
    parent2 = selection_tournament(population, tournament_size)

    # Apply crossover with the given probability
    if np.random.random() < crossover_prob:
        child = crossover(parent1, parent2)

        # Apply mutation with the given probability
        if np.random.random() < mutation_prob:
            mutate_random_strategy(child, terminals)

        # Evaluate the child's fitness
        child.fitness = evaluate_fitness(child, data, true_labels)

        return (child, (parent1, parent2))
    else:
        return None
