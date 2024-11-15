#!/usr/bin/env python3

# Filename: simulation.py
# Created on: November 13, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import time
import logging
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from datetime import datetime

from .gene import Gene
from .operators import generate_child
from .parameters import (
    LOG_FOLDER,
    LOG_PREFIX,
    BREAST_CANCER_TRAIN_DATASET,
    POPULATION_SIZE,
    NUM_GENERATIONS,
    CROSSOVER_PROB,
    MUTATION_PROB,
    ELITISM_SIZE,
    THREAD_INITIAL_SEED
)
from .population import (
    generate_initial_population,
    count_duplicated_genes,
    evaluate_fitness,
)
from .utils import read_data_from_csv, print_line


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

    def _load_data(self):
        labeled_data = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET)

        self.true_labels = [label for label, _ in labeled_data]
        self.data = [features for _, features in labeled_data]

    def evaluate_generation(self, generation, population):
        """
        Evaluate the fitness of the population and log the stats.
        """
        median_fitness = np.median([gene.fitness for gene in population])
        mean_fitness = np.mean([gene.fitness for gene in population])
        std_fitness = np.std([gene.fitness for gene in population])
        best_gene = max(population)
        worst_gene = min(population)

        self.logger.info(f"Stats for Generation {generation}:")
        self.logger.info(f"\tBest:   {best_gene.fitness}")
        self.logger.info(f"\tWorst:  {worst_gene.fitness}")
        self.logger.info(f"\tMean:   {mean_fitness}")
        self.logger.info(f"\tMedian: {median_fitness}")
        self.logger.info(f"\tStd:    {std_fitness}")

        print(f"Stats for Generation {generation}:")
        print(f"\tBest:   {best_gene.fitness}")
        print(f"\tWorst:  {worst_gene.fitness}")
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
                new_genes[i].fitness = evaluate_fitness(
                    new_genes[i], self.data, self.true_labels
                )

                population[index] = new_genes[i]

            total_replaced_genes += duplicated_genes

        print(f"Replaced {total_replaced_genes} duplicated genes")

        return total_replaced_genes

    def evaluate_population(self, population, workers=None):
        """
        Evaluate the fitness of the population using a multi-processing approach

        @param population: The population to evaluate
        @param workers: The number of workers to use. If None, uses the default number
                        of workers (all)
        """
        with (
            ProcessPoolExecutor(max_workers=workers)
            if workers
            else ProcessPoolExecutor()
        ) as executor:
            futures = {
                executor.submit(
                    evaluate_fitness, gene, self.data, self.true_labels
                ): gene
                for gene in population
            }

            for future in futures:
                gene = futures[future]
                gene.fitness = future.result()

    def generate_childs(
            self, population, seed_sequence, workers=None, total_possible_crossovers=POPULATION_SIZE
    ) -> Tuple[List[Tuple[Gene, Tuple[Gene, Gene]]], int]:
        """
        Generates the next generation of a given population in parallel.

        @param population: The current population
        @param total_possible_crossovers: The total number of possible crossovers
        @param workers: The number of workers to use. If None, uses the default number
                        of workers (all)
        @return: A list of tuples (child, (parent1, parent2)) for the new generation
        """
        results = []

        with (
            ProcessPoolExecutor(max_workers=workers)
            if workers
            else ProcessPoolExecutor()
        ) as executor:
            futures = []
            for _ in range(total_possible_crossovers):
                # Pass a unique seed to each process and increment for the next
                seed_sequence += 1

                futures.append(
                    executor.submit(
                        generate_child,
                        population,
                        self.data,
                        self.true_labels,
                        CROSSOVER_PROB,
                        MUTATION_PROB,
                        seed_sequence
                    )
                )

            # Collect results, filtering out any None values where crossover did not occur
            results.extend(
                future.result() for future in futures if future.result() is not None
            )

        return results, seed_sequence

    def run(self):
        self._load_data()

        if self.data is None or self.true_labels is None:
            raise ValueError("Data or true labels not loaded")

        sim_time = time.time()
        population = generate_initial_population(POPULATION_SIZE, "half_and_half")

        self.handle_duplicate_genes(population)

        self.evaluate_population(population, 2)

        seed_sequence = THREAD_INITIAL_SEED

        for generation in range(1, NUM_GENERATIONS + 1):
            population_size = len(population)

            self.logger.info(f"Generation {generation}")
            self.logger.info(f"Population size: {population_size}")
            print(f"Generation {generation}:")
            print(f"Population size: {population_size}")

            start_time = time.time()

            # Elitism: Keep the best ELITISM_SIZE genes from the previous generation
            best_genes = sorted(population)[-ELITISM_SIZE:]

            childs, seed_sequence = self.generate_childs(population, seed_sequence, 2)

            # Add the child to the new generation only if it is better than your parents
            # and remove the worst parent
            for child, (selected1, selected2) in childs:
                worst_parent = min(selected1, selected2)

                if child > worst_parent:
                    if worst_parent in population:
                        # Check if the worst parent is in the population
                        # (In some cases, it may have already been replaced by another child)
                        index_to_replace = population.index(worst_parent)
                        population[index_to_replace] = child
                    else:
                        # If the worst parent is not in the population, just add the child
                        population.append(child)
                        pass

            # Add the best genes from the previous generation to the new generation
            for best_gene in best_genes:
                if best_gene not in population:
                    population.append(best_gene)

            # Ensure the population size remains consistent by keeping only
            # the best POPULATION_SIZE genes (which are now at the end of the list)
            population = population[-POPULATION_SIZE:]

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
