#!/usr/bin/env python3

# Filename: population.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

def generate_random_tree(depth):
    # Gera uma árvore de expressão aleatória até a profundidade `depth`.
    pass

def generate_initial_population(population_size):
    return [Gene(generate_random_tree(TREE_MAX_DEPTH)) for _ in range(population_size)]
