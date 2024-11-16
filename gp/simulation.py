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
    N_BEST,
    TREE_MAX_DEPTH,
    TREE_MIN_DEPTH,
    DATA_DIMENSION,
    NON_TERMINAL_PROB,
    TERMINAL_PROB,
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
from .utils import print_tree, read_data_from_csv, print_line


class Simulation:
    def __init__(
        self,
        seed,
        thread_initial_seed,
        population_size,
        crossovers_by_generation,
        num_generations,
        crossover_prob,
        mutation_prob,
        tournament_size,
        elitism_size,
        simulation_id=None,
    ):
        self.train_data = None
        self.train_true_labels = None
        self.test_data = None
        self.test_true_labels = None

        # Simulation parameters
        self.seed = seed
        self.thread_initial_seed = thread_initial_seed
        self.population_size = population_size
        self.crossovers_by_generation = crossovers_by_generation
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

        np.random.seed(self.seed)  # Seed the random number generator

        if simulation_id is None:
            now = datetime.now()

            self.raw_log_filename = (
                f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{LOG_FILE_SUFFIX}.dat"
            )

            self.log_filename = (
                f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{LOG_FILE_SUFFIX}.log"
            )

            self.json_filename = f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{REPRODUCIBILITY_FILE_SUFFIX}.json"

        else:
            self.raw_log_filename = f"{LOG_FOLDER}{simulation_id}_{LOG_FILE_SUFFIX}.dat"
            self.log_filename = f"{LOG_FOLDER}{simulation_id}_{LOG_FILE_SUFFIX}.log"
            self.json_filename = (
                f"{LOG_FOLDER}{simulation_id}_{REPRODUCIBILITY_FILE_SUFFIX}.json"
            )

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
                "Generation|Population|DuplicatedGenes|BestFitness|WorstFitness|MeanFitness|MedianFitness|StdFitness|TimeTaken\n"
            )

    def _write_raw_log(self, message):
        with open(self.raw_log_filename, "a") as f:
            f.write(message)

    def _write_log(self, message, level=logging.INFO, print_message=True):
        self.logger.log(level, message)

        if print_message:
            print(message)

    def _load_data(self):
        labeled_train_data = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET)
        labeled_test_data = read_data_from_csv(BREAST_CANCER_TEST_DATASET)

        self.train_true_labels = [label for label, _ in labeled_train_data]
        self.train_data = [features for _, features in labeled_train_data]

        self.test_true_labels = [label for label, _ in labeled_test_data]
        self.test_data = [features for _, features in labeled_test_data]

        match NORMALIZATION_METHOD:
            case NormalizationMethod.MIN_MAX:
                scaler = MinMaxScaler()
                self.train_data = scaler.fit_transform(self.train_data)
                self.test_data = scaler.transform(self.test_data)

            case NormalizationMethod.STANDARD:
                scaler = StandardScaler()
                self.train_data = scaler.fit_transform(self.train_data)
                self.test_data = scaler.transform(self.test_data)

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

        duplicated = count_duplicated_genes(population)[0]

        # Log the stats
        self._write_log(
            f"Stats:\n"
            f"|> Best:       {best_gene.fitness}\n"
            f"|> Worst:      {worst_gene.fitness}\n"
            f"|> Mean:       {mean_fitness}\n"
            f"|> Median:     {median_fitness}\n"
            f"|> Std:        {std_fitness}\n"
            f"|> Duplicated: {duplicated}"
        )

        if time_taken is not None:
            self._write_log(f"Generation took {time_taken:.2f} s")

        # Write the raw log
        self._write_raw_log(
            f"{generation_number}|{len(population)}|{duplicated}|{best_gene.fitness}|"
            f"{worst_gene.fitness}|{mean_fitness}|{median_fitness}|"
            f"{std_fitness}|{time_taken or 'N/A'}\n"
        )

    def get_unique_best(self, n_best_fitness, population) -> List[Gene]:
        """
        Get the best genes from the population, removing duplicates

        @param n_best_fitness: The number of best fitness to keep
        @param population: The population to get the best genes
        @return: The best genes found
        """
        seen_fitness = set()
        unique_best = []

        for gene in heapq.nlargest(
            len(population), population, key=lambda g: g.fitness
        ):
            if gene.fitness not in seen_fitness:
                unique_best.append(gene)
                seen_fitness.add(gene.fitness)
            if len(unique_best) == n_best_fitness:
                break

        return unique_best

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
                    new_genes[i], self.train_data, self.train_true_labels
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
                    evaluate_fitness, gene, self.train_data, self.train_true_labels
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
                        self.train_data,
                        self.train_true_labels,
                        self.tournament_size,
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
            "POPULATION_SIZE": self.population_size,
            "CROSSOVERS_BY_GENERATION": self.crossovers_by_generation,
            "NUM_GENERATIONS": self.num_generations,
            "CROSSOVER_PROB": self.crossover_prob,
            "MUTATION_PROB": self.mutation_prob,
            "TOURNAMENT_SIZE": self.tournament_size,
            "ELITISM_SIZE": self.elitism_size,
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

        if self.train_data is None or self.train_true_labels is None:
            raise ValueError("Data or true labels not loaded")

        sim_time = time.time()
        population = generate_initial_population(self.population_size, "half_and_half")

        self.handle_duplicate_genes(population)

        self.evaluate_population(population, workers=multi_processing_workers)

        # Ensure all genes are within the height bounds and have a fitness value
        for gene in population:
            assert (
                gene.height <= TREE_MAX_DEPTH and gene.height >= TREE_MIN_DEPTH
            ), f"Gene height is out of bounds [{TREE_MIN_DEPTH}, {TREE_MAX_DEPTH}]: {gene.height}"

            assert gene.fitness is not None, "Gene fitness is None"

        # Evaluate the initial population
        best_genes = self._run_evolution(
            population, N_BEST, multi_processing_workers=multi_processing_workers
        )

        self._run_test(best_genes)

        total_sim_time = time.time() - sim_time
        self._write_log(f"Simulation took {total_sim_time:.2f} s")

    def _run_evolution(
        self, population, n_best_fitness, multi_processing_workers=None
    ) -> List[Gene]:
        """
        Run the evolution process

        @param population: The initial population
        @param n_best_fitness: The number of best fitness to keep
        @param multi_processing_workers: The number of workers to use in the multi-processing
        @return: The best genes found
        """

        # Seed sequence to ensure unique seeds for each process
        seed_sequence = self.thread_initial_seed

        generation_times = []

        for generation in range(1, self.num_generations + 1):
            self._write_log(f"Generation: {generation}")
            self._write_log(f"Population size: {len(population)}")

            start_time = time.time()

            # Elitism: Keep the best ELITISM_SIZE genes from the previous generation
            new_population = heapq.nlargest(self.elitism_size, population)

            # Get the childs and the new seed sequence
            childs, seed_sequence = self.generate_childs(
                population,
                seed_sequence,
                self.crossovers_by_generation,
                self.crossover_prob,
                self.mutation_prob,
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
            remaining_needed = self.population_size - len(new_population)

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

        return self.get_unique_best(n_best_fitness, population)

    def _run_test(self, best_genes):
        """
        Run the test phase using the best gene found

        @param best_gene: The best genes found
        """
        self._write_raw_log("-" * 200 + "\n")
        self._write_raw_log("Ranking|TestFitness|TrainFitness|GeneTreeHeight\n")

        for i, gene in enumerate(best_genes, start=1):
            # Evaluate the test data
            test_fitness = evaluate_fitness(gene, self.test_data, self.test_true_labels)

            self._write_log(f"Top {i} gene:")
            self._write_log(f"|> Test fitness:  {test_fitness}")
            self._write_log(f"|> Train fitness: {gene.fitness}")
            self._write_log(f"|> Tree height:   {gene.height}")

            self._write_raw_log(f"{i}|{test_fitness}|{gene.fitness}|{gene.height}\n")
