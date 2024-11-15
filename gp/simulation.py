#!/usr/bin/env python3

# Filename: simulation.py
# Created on: November 13, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import json
import heapq
import time
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from datetime import datetime

from .gene import Gene
from .operators import generate_child

from .parameters import (
    TREE_MAX_DEPTH,
    TREE_MIN_DEPTH,
    DATA_DIMENSION,
    NON_TERMINAL_PROB,
    TERMINAL_PROB,
    POPULATION_SIZE,
    CROSSOVERS_BY_GENERATION,
    NUM_GENERATIONS,
    CROSSOVER_PROB,
    MUTATION_PROB,
    TOURNAMENT_SIZE,
    ELITISM_SIZE,
    NORMALIZATION_METHOD,
    NON_TERMINAL,
    TERMINAL,
    NormalizationMethod,
)

from .constants import (
    BREAST_CANCER_TEST_DATASET,
    BREAST_CANCER_TRAIN_DATASET,
    LOG_FOLDER,
    LOG_FILE_SUFFIX,
    REPRODUCIBILITY_FILE_SUFFIX,
)

from .population import (
    generate_initial_population,
    count_duplicated_genes,
    evaluate_fitness,
)
from .utils import read_data_from_csv, print_line


class Simulation:
    def __init__(
        self,
        seed=np.random.randint(0, 2**32 - 1),
        thread_initial_seed=np.random.randint(0, 2**32 - 1),
    ):
        self.data = None
        self.true_labels = None
        self.seed = seed
        self.thread_initial_seed = thread_initial_seed

        np.random.seed(self.seed)  # Seed the random number generator

        now = datetime.now()

        self.raw_log_filename = (
            f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{LOG_FILE_SUFFIX}.dat"
        )

        self.log_filename = (
            f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{LOG_FILE_SUFFIX}.log"
        )

        self.json_filename = f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{REPRODUCIBILITY_FILE_SUFFIX}.json"

        logging.basicConfig(
            filename=self.log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

    def _initialize_raw_log_file(self):
        """
        Initializes the log file with headers for better readability.
        """
        with open(self.raw_log_filename, "w") as f:
            f.write(
                "Generation\tPopulation\tBestFitness\tWorstFitness\tMeanFitness\tMedianFitness\tStdFitness\tTimeTaken\n"
            )

    def _write_raw_log(self, message):
        with open(self.raw_log_filename, "a") as f:
            f.write(message)

    def _write_log(self, message, level=logging.INFO, print_message=True):
        self.logger.log(level, message)

        if print_message:
            print(message)

    def _load_data(self):
        labeled_data = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET)

        self.true_labels = [label for label, _ in labeled_data]
        self.data = [features for _, features in labeled_data]

        match NORMALIZATION_METHOD:
            case NormalizationMethod.MIN_MAX:
                scaler = MinMaxScaler()
                self.data = scaler.fit_transform(self.data)

            case NormalizationMethod.STANDARD:
                scaler = StandardScaler()
                self.data = scaler.fit_transform(self.data)

            case _:
                raise ValueError("Invalid normalization method")

    def evaluate_generation(self, population, generation_number, time_taken=None):
        """
        Evaluate the fitness of the population and log the stats.
        """
        median_fitness = np.median([gene.fitness for gene in population])
        mean_fitness = np.mean([gene.fitness for gene in population])
        std_fitness = np.std([gene.fitness for gene in population])
        best_gene = max(population)
        worst_gene = min(population)

        # Log the stats
        self._write_log(
            f"Stats:\n\tBest:   {best_gene.fitness}\n\tWorst:  {worst_gene.fitness}\n\tMean:   {mean_fitness}\n\tMedian: {median_fitness}\n\tStd:    {std_fitness}"
        )

        if time_taken is not None:
            self._write_log(f"Generation took {time_taken:.2f} s")

        # Write the raw log
        self._write_raw_log(
            f"{generation_number}\t{len(population)}\t{best_gene.fitness}\t{worst_gene.fitness}\t{mean_fitness}\t{median_fitness}\t{std_fitness}\t{time_taken or 'N/A'}\n"
        )

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

        self._write_log(f"Replaced {total_replaced_genes} duplicated genes")

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
        total_possible_crossovers,
        cross_prob,
        mut_prob,
        workers=None,
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
                        cross_prob,
                        mut_prob,
                        seed_sequence,
                    )
                )

            # Collect results, filtering out any None values where crossover did not occur
            childs.extend(
                future.result() for future in futures if future.result() is not None
            )

        return childs, seed_sequence

    def save_simulation_parameters(self):
        """
        Save the simulation parameters to the json file
        """
        data_to_save = {
            "SEED": self.seed,
            "THREAD_INITIAL_SEED": self.thread_initial_seed,
            "TREE_MAX_DEPTH": TREE_MAX_DEPTH,
            "TREE_MIN_DEPTH": TREE_MIN_DEPTH,
            "DATA_DIMENSION": DATA_DIMENSION,
            "NON_TERMINAL_PROB": NON_TERMINAL_PROB,
            "TERMINAL_PROB": TERMINAL_PROB,
            "POPULATION_SIZE": POPULATION_SIZE,
            "CROSSOVERS_BY_GENERATION": CROSSOVERS_BY_GENERATION,
            "NUM_GENERATIONS": NUM_GENERATIONS,
            "CROSSOVER_PROB": CROSSOVER_PROB,
            "MUTATION_PROB": MUTATION_PROB,
            "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
            "ELITISM_SIZE": ELITISM_SIZE,
            "NORMALIZATION_METHOD": str(NORMALIZATION_METHOD),
            "NON_TERMINAL": NON_TERMINAL,
            "TERMINAL": TERMINAL,
            "BREAST_CANCER_TEST_DATASET": BREAST_CANCER_TEST_DATASET,
            "BREAST_CANCER_TRAIN_DATASET": BREAST_CANCER_TRAIN_DATASET,
        }

        with open(self.json_filename, "w") as json_file:
            json.dump(data_to_save, json_file, indent=4)

    def run(self, multi_processing_workers=None):
        self._load_data()
        self._initialize_raw_log_file()

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
        self._write_log(f"Simulation took {total_sim_time:.2f} s")

    def _run_evolution(self, population, multi_processing_workers=None):
        # Seed sequence to ensure unique seeds for each process
        seed_sequence = self.thread_initial_seed

        generation_times = []

        for generation in range(1, NUM_GENERATIONS + 1):
            self._write_log(f"Generation: {generation}")
            self._write_log(f"Population size: {len(population)}")

            start_time = time.time()

            # Elitism: Keep the best ELITISM_SIZE genes from the previous generation
            new_population = heapq.nlargest(ELITISM_SIZE, population)

            # Get the childs and the new seed sequence
            childs, seed_sequence = self.generate_childs(
                population,
                seed_sequence,
                CROSSOVERS_BY_GENERATION,
                CROSSOVER_PROB,
                MUTATION_PROB,
                workers=multi_processing_workers,
            )

            self._write_log(f"Generated childs: {len(childs)}")

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

            time_taken = time.time() - start_time
            generation_times.append(time_taken)

            self.evaluate_generation(population, generation, time_taken)

            print_line()

        self._write_log(f"Avg. generation time: {np.mean(generation_times):.2f} s")
        self._write_log(f"Normalization method: {NORMALIZATION_METHOD}")
        self._write_log(f"Finished evolution")
