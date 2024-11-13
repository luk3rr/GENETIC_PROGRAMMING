#!/usr/bin/env python3

# Filename: population.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import random

from typing import List

from .gene import *
from .parameters import *


def generate_random_tree(depth) -> Node:
    """
    Generate a random tree with the given depth

    @param depth: The depth of the tree
    @return: A random tree
    """
    if depth == 0:
        return Node(random.choice(TERMINAL), depth=TREE_MAX_DEPTH)

    if random.random() < TERMINAL_PROB:
        return Node(random.choice(TERMINAL), depth=TREE_MAX_DEPTH - depth)

    left = generate_random_tree(depth - 1)
    right = generate_random_tree(depth - 1)

    return Node(
        random.choice(NON_TERMINAL),
        depth=TREE_MAX_DEPTH - depth,
        left=left,
        right=right,
    )


def generate_initial_population(population_size) -> List[Gene]:
    """
    Generate the initial population of the genetic programming algorithm

    @param population_size: The size of the population
    @return: A list with the initial population
    """
    return [Gene(generate_random_tree(TREE_MAX_DEPTH)) for _ in range(population_size)]
