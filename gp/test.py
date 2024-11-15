#!/usr/bin/env python3

# Filename: test.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from .gene import Gene
from .parameters import *
from .population import *
from .operators import *
from .utils import *


class GPTest:
    def __init__(self):
        pass

    def test_all(self):
        print_line()
        self.test_tree_generation()
        print_line()
        self.test_crossover()
        print_line()
        self.test_one_point_mutation()
        print_line()
        self.test_expand_mutation()
        print_line()
        self.test_shrink_mutation()
        print_line()
        self.test_generate_initial_population_grow()
        print_line()
        self.test_generate_initial_population_full()
        print_line()
        self.test_generate_initial_population_half_and_half()

    def test_evaluate_fitness(self):
        pass

    def test_crossover(self):
        print("Running crossover test...")

        parent1 = Gene(generate_random_tree(full, TREE_MAX_DEPTH))
        parent2 = Gene(generate_random_tree(full, TREE_MAX_DEPTH))

        print("Before crossover:")
        print("Parent 1:", parent1.show_prefix())
        print("Parent 2:", parent2.show_prefix())

        print("\nParent 1 tree")
        print_tree(parent1.root_node)

        print("\nParent 2 tree")
        print_tree(parent2.root_node)

        child = crossover(parent1, parent2)

        print("\nChild tree")
        print_tree(child.root_node)

        print("\nAfter crossover:")
        print("Child:", child.show_prefix())

    def test_one_point_mutation(self):
        print("Running one-point mutation test...")

        gene = Gene(generate_random_tree(full, TREE_MAX_DEPTH))

        print("Before mutation:")
        print(gene.show_prefix())

        mutate(gene, strategy=one_point_mutation)

        print("\nAfter mutation:")
        print(gene.show_prefix())

    def test_expand_mutation(self):
        print("Running expand mutation test...")

        gene = Gene(generate_random_tree(full, TREE_MAX_DEPTH))

        print("Before expand mutation:")
        print(gene.show_prefix())

        mutate(gene, strategy=expand_mutation)

        print("\nAfter expand mutation:")
        print(gene.show_prefix())

    def test_shrink_mutation(self):
        print("Running shrink mutation test...")

        gene = Gene(generate_random_tree(full, TREE_MAX_DEPTH))

        print("Before shrink mutation:")
        print(gene.show_prefix())

        mutate(gene, strategy=shrink_mutation)

        print("\nAfter shrink mutation:")
        print(gene.show_prefix())

    def test_tree_generation(self, num_trees=5):
        print(
            f"Generating {num_trees} random expression trees with max depth {TREE_MAX_DEPTH}:\n"
        )

        for i in range(num_trees):
            gene = Gene(generate_random_tree(full, TREE_MAX_DEPTH))

            print(f"Tree {i + 1}:")
            print("Prefix notation:", gene.show_prefix())
            print("Infix notation:", gene.show_infix())

    def test_generate_initial_population_grow(self):
        self._test_generate_initial_population("grow")

    def test_generate_initial_population_full(self):
        self._test_generate_initial_population("full")

    def test_generate_initial_population_half_and_half(self):
        self._test_generate_initial_population("half_and_half")

    def _test_generate_initial_population(self, strategy):
        print("Generating initial population with strategy: ", strategy)

        population = generate_initial_population(POPULATION_SIZE, strategy)

        for i, gene in enumerate(population):
            print(f"Gene {i + 1}:")
            print("Prefix notation:", gene.show_prefix())
            print("Infix notation:", gene.show_infix())
            print("Tree:\n")
            print_tree(gene.root_node)
