#!/usr/bin/env python3

# Filename: simulation.py
# Created on: November 13, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import random
import time
import logging
import numpy as np

from datetime import datetime

from .gene import *
from .population import *
from .parameters import *
from .operators import *


class Simulation:
    def __init__(self):
        self.data = None
        self.true_labels = None

        now = datetime.now()

        log_filename = (
            f"{LOG_FOLDER}{LOG_PREFIX}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

    def evaluate_generation(self, generation, population):
        """
        Evaluate the fitness of the population and log the stats.
        """
        median_fitness = np.median([gene.fitness for gene in population])
        mean_fitness = np.mean([gene.fitness for gene in population])
        std_fitness = np.std([gene.fitness for gene in population])
        best_gene = max(population)

        self.logger.info(f"Stats for Generation {generation}:")
        self.logger.info(f"\tBest:   {best_gene.fitness}")
        self.logger.info(f"\tMean:   {mean_fitness}")
        self.logger.info(f"\tMedian: {median_fitness}")
        self.logger.info(f"\tStd:    {std_fitness}")

        print(f"Stats for Generation {generation}:")
        print(f"\tBest:   {best_gene.fitness}")
        print(f"\tMean:   {mean_fitness}")
        print(f"\tMedian: {median_fitness}")
        print(f"\tStd:    {std_fitness}")

    def handle_duplicate_genes(self, population):
        """
        Handle duplicated genes in the population by generating new genes
        to replace the duplicates

        @param population: The population to handle
        @return: The number of replaced genes
        """
        total_replaced_genes = 0

        while True:
            duplicated_genes, duplicated_indexes = count_duplicated_genes(population)

            if duplicated_genes == 0:
                # No duplicated genes found, break the loop
                break

            new_genes = generate_initial_population(duplicated_genes, "half_and_half")


            if len(new_genes) != duplicated_genes:
                raise ValueError(
                    f"Expected {duplicated_genes} new genes, but got {len(new_genes)}"
                )

            for i, index in enumerate(duplicated_indexes):
                new_genes[i].evaluate_fitness(self.data, self.true_labels)

                population[index] = new_genes[i]

            total_replaced_genes += duplicated_genes

        return total_replaced_genes

    def load_data(self):
        labeled_data = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET)

        self.true_labels = [label for label, _ in labeled_data]
        self.data = [features for _, features in labeled_data]


    def run(self):
        self.load_data()

        if self.data is None or self.true_labels is None:
            raise ValueError("Data or true labels not loaded")

        sim_time = time.time()
        population = generate_initial_population(POPULATION_SIZE, "half_and_half")

        for generation in range(1, NUM_GENERATIONS + 1):
            population_size = len(population)

            self.logger.info(f"Generation {generation}")
            self.logger.info(f"Population size: {population_size}")
            print(f"Generation {generation}:")
            print(f"Population size: {population_size}")

            start_time = time.time()

            for gene in population:
                gene.evaluate_fitness(self.data, self.true_labels)

            # Elitism: Keep the best N genes from the previous generation
            best_genes = sorted(population, reverse=True)[:ELITISM_SIZE]

            for _ in range(POPULATION_SIZE):
                # Select the genes for the next generation
                selected1 = selection_tournament(population)
                selected2 = selection_tournament(population)

                child_generated = False

                if random.random() < CROSSOVER_PROB:
                    child_generated = True
                    child = crossover(selected1, selected2)

                    # Mutate the child
                    if random.random() < MUTATION_PROB:
                        mutate_random_strategy(child)

                    child.evaluate_fitness(self.data, self.true_labels)

                    # Add child to end of population
                    # Ensures that if the population is truncated, the child will not be lost
                    population.append(child)

            # Add the best genes from the previous generation to the new generation
            for best_gene in best_genes:
                if best_gene not in population:
                    population.append(best_gene)

            # Make sure the population size is consistent
            population = population[:POPULATION_SIZE]

            duplicated_genes = self.handle_duplicate_genes(population)
            self.logger.info(f"Replaced duplicated genes: {duplicated_genes}")
            print(f"Replaced duplicated genes: {duplicated_genes}")

            self.evaluate_generation(generation, population)

            end_time = time.time()
            time_taken = end_time - start_time

            self.logger.info(f"Generation {generation} took {time_taken:.2f} seconds")
            print(f"Generation {generation} took {time_taken:.2f} seconds")
            print_line()

        sim_end_time = time.time()
        total_sim_time = sim_end_time - sim_time
        self.logger.info(f"Simulation took {total_sim_time:.2f} seconds")
        print(f"Simulation took {total_sim_time:.2f} seconds")
