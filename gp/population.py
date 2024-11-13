#!/usr/bin/env python3

# Filename: population.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import random

from typing import List

from .gene import *
from .parameters import *


def grow(depth) -> Node:
    """
    Grow strategy for the genetic programming algorithm

    @param depth: The depth of the tree
    @return: A random node
    """
    if depth == 0:
        return Node(random.choice(TERMINAL), depth=TREE_MAX_DEPTH)

    if random.random() < TERMINAL_PROB:
        return Node(random.choice(TERMINAL), depth=TREE_MAX_DEPTH - depth)

    left = grow(depth - 1)
    right = grow(depth - 1)

    return Node(
        random.choice(NON_TERMINAL),
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
        return Node(random.choice(TERMINAL), depth=TREE_MAX_DEPTH)

    left = full(depth - 1)
    right = full(depth - 1)

    return Node(
        random.choice(NON_TERMINAL),
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
    return grow(depth) if random.random() < 0.5 else full(depth)


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

        population = []

        for depth in range(1, TREE_MAX_DEPTH):
            population += [
                Gene(generate_random_tree(depth, grow))
                for _ in range(genes_per_depth // 2)
            ] + [
                Gene(generate_random_tree(depth, full))
                for _ in range(genes_per_depth // 2)
            ]

        return population
    else:
        raise ValueError("Invalid strategy")
