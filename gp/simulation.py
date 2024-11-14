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
        now = datetime.now()
        log_filename = f"log_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

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

    def run(self):
        sim_time = time.time()
        labeled_data = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET)

        true_labels = [label for label, _ in labeled_data]
        data = [features for _, features in labeled_data]

        for generation in range(1, NUM_GENERATIONS + 1):
            self.logger.info(f"Generation {generation}:")
            print(f"Generation {generation}:")

            start_time = time.time()

            population = generate_initial_population(POPULATION_SIZE, "half_and_half")

            for gene in population:
                gene.evaluate_fitness(data, true_labels)

            # Elitism: Keep the best N genes from the previous generation
            best_genes = sorted(population, reverse=True)[:ELITISM_SIZE]

            for _ in range(POPULATION_SIZE // 2):
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

                    child.evaluate_fitness(data, true_labels)

                    population.append(child)

                # Mutate the parents
                if random.random() < MUTATION_PROB:
                    mutate_random_strategy(selected1)
                    selected1.evaluate_fitness(data, true_labels)

                if random.random() < MUTATION_PROB:
                    mutate_random_strategy(selected2)
                    selected2.evaluate_fitness(data, true_labels)

                # If child_generated, then remove the parent with the worst fitness
                if child_generated:
                    population.remove(min(selected1, selected2))

            # After generating children and applying mutations, we now keep the best genes
            population = best_genes + population

            # Make sure the population size is consistent
            population = population[:POPULATION_SIZE]

            self.evaluate_generation(generation, population)

            end_time = time.time()
            time_taken = end_time - start_time
            self.logger.info(f"Generation {generation} took {time_taken:.2f} seconds")
            print(
                f"Generation {generation} took {time_taken:.2f} seconds"
            )

        sim_end_time = time.time()
        total_sim_time = sim_end_time - sim_time
        self.logger.info(f"Simulation took {total_sim_time:.2f} seconds")
        print(
            f"Simulation took {total_sim_time:.2f} seconds"
        )
