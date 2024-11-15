#!/usr/bin/env python3

# Filename: simulation.py
# Created on: November 13, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import heapq
import time
import logging
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from datetime import datetime

from .gene import Gene
from .operators import generate_child
from .parameters import (
    CROSSOVERS_BY_GENERATION,
    LOG_FOLDER,
    LOG_PREFIX,
    BREAST_CANCER_TRAIN_DATASET,
    POPULATION_SIZE,
    NUM_GENERATIONS,
    CROSSOVER_PROB,
    MUTATION_PROB,
    ELITISM_SIZE,
    THREAD_INITIAL_SEED,
    TREE_MAX_DEPTH,
    TREE_MIN_DEPTH,
)
from .population import (
    generate_initial_population,
    count_duplicated_genes,
    evaluate_fitness,
)
from .utils import print_tree, read_data_from_csv, print_line


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

    def evaluate_generation(self, population):
        """
        Evaluate the fitness of the population and log the stats.
        """
        median_fitness = np.median([gene.fitness for gene in population])
        mean_fitness = np.mean([gene.fitness for gene in population])
        std_fitness = np.std([gene.fitness for gene in population])
        best_gene = max(population)
        worst_gene = min(population)

        self.logger.info(f"Stats:")
        self.logger.info(f"\tBest:   {best_gene.fitness}")
        self.logger.info(f"\tWorst:  {worst_gene.fitness}")
        self.logger.info(f"\tMean:   {mean_fitness}")
        self.logger.info(f"\tMedian: {median_fitness}")
        self.logger.info(f"\tStd:    {std_fitness}")

        print(f"Stats:")
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
        with ProcessPoolExecutor(max_workers=workers) as executor:
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
        self,
        population,
        seed_sequence,
        workers=None,
        total_possible_crossovers=CROSSOVERS_BY_GENERATION,
    ) -> Tuple[List[Tuple[Gene, Tuple[Gene, Gene]]], int]:
        """
        Generates the next generation of a given population in parallel.

        @param population: The current population
        @param total_possible_crossovers: The total number of possible crossovers
        @param workers: The number of workers to use. If None, uses the default number
                        of workers (all)
        @return: A list of tuples (child, (parent1, parent2)) for the new generation
        """
        childs = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
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
                        seed_sequence,
                    )
                )

            # Collect results, filtering out any None values where crossover did not occur
            childs.extend(
                future.result() for future in futures if future.result() is not None
            )

        return childs, seed_sequence

    def run(self, multi_processing_workers=None):
        self._load_data()

        if self.data is None or self.true_labels is None:
            raise ValueError("Data or true labels not loaded")

        sim_time = time.time()
        population = generate_initial_population(POPULATION_SIZE, "half_and_half")

        self.handle_duplicate_genes(population)

        self.evaluate_population(population, workers=multi_processing_workers)

        # Ensure all genes are within the height bounds and have a fitness value
        for gene in population:
            assert (
                gene.height <= TREE_MAX_DEPTH and gene.height >= TREE_MIN_DEPTH
            ), f"Gene height is out of bounds [{TREE_MIN_DEPTH}, {TREE_MAX_DEPTH}]: {gene.height}"

            assert gene.fitness is not None, "Gene fitness is None"

        # Evaluate the initial population
        self._run_evolution(
            population, multi_processing_workers=multi_processing_workers
        )

        total_sim_time = time.time() - sim_time
        self.logger.info(f"Simulation took {total_sim_time:.2f} seconds")
        print(f"Simulation took {total_sim_time:.2f} seconds")

    def _run_evolution(self, population, multi_processing_workers=None):
        # Seed sequence to ensure unique seeds for each process
        seed_sequence = THREAD_INITIAL_SEED

        for generation in range(1, NUM_GENERATIONS + 1):
            self.logger.info(f"Generation: {generation}")
            print(f"Generation: {generation}")
            print(f"Population size: {len(population)}")

            start_time = time.time()

            # Elitism: Keep the best ELITISM_SIZE genes from the previous generation
            new_population = heapq.nlargest(ELITISM_SIZE, population)

            # Get the childs and the new seed sequence
            childs, seed_sequence = self.generate_childs(
                population, seed_sequence, workers=multi_processing_workers
            )

            print(f"Generated childs: {len(childs)}")
            self.logger.info(f"Generated childs: {len(childs)}")

            replaced_genes = set()

            # Add the child to the new generation only if it is better than its parents
            # and remove the worst parent
            for child, (selected1, selected2) in childs:
                worst_parent = min(selected1, selected2)
                if child > worst_parent and worst_parent not in replaced_genes:
                    new_population.append(child)
                    replaced_genes.add(worst_parent)

            # Ensure we select the correct number of additional genes to reach POPULATION_SIZE
            remaining_needed = POPULATION_SIZE - len(new_population)

            # Get the genes that were not replaced
            additional_genes = [g for g in population if g not in new_population]

            # Add the best remaining genes to complete the population size
            new_population.extend(heapq.nlargest(remaining_needed, additional_genes))

            population = new_population

            self.evaluate_generation(population)

            time_taken = time.time() - start_time
            self.logger.info(f"Generation {generation} took {time_taken:.2f} seconds")
            print(f"Generation {generation} took {time_taken:.2f} seconds")
            print_line()
