#!/usr/bin/env python3

# Filename: simulation.py
# Created on: November 13, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import json
import heapq
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from datetime import datetime

from .gene import Gene
from .operators import generate_child

from .parameters import (
    TERMINAL_BASE,
    TREE_MAX_DEPTH,
    TREE_MIN_DEPTH,
    NON_TERMINAL_PROB,
    TERMINAL_PROB,
    NORMALIZATION_METHOD,
    NON_TERMINAL,
    NormalizationMethod,
)

from .constants import (
    BREAST_CANCER_TEST_DATASET,
    BREAST_CANCER_TRAIN_DATASET,
    LOG_FOLDER,
    LOG_FILE_SUFFIX,
    REPRODUCIBILITY_FILE_SUFFIX,
    WINE_RED_TRAIN_DATASET,
)

from .population import (
    generate_initial_population,
    count_duplicated_genes,
    evaluate_fitness,
)
from .utils import read_data_from_csv


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
        dataset,
        simulation_id=None,
    ):
        # Data
        self.dataset = dataset
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

        self.terminals = None

        if simulation_id is None:
            now = datetime.now()

            self.raw_log_filename = (
                f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{LOG_FILE_SUFFIX}.dat"
            )

            self.json_filename = f"{LOG_FOLDER}{now.strftime('%Y-%m-%d_%H-%M-%S')}_{REPRODUCIBILITY_FILE_SUFFIX}.json"

        else:
            simulation_id = simulation_id.strip().replace(" ", "_")

            self.raw_log_filename = f"{LOG_FOLDER}{simulation_id}_{LOG_FILE_SUFFIX}.dat"
            self.json_filename = (
                f"{LOG_FOLDER}{simulation_id}_{REPRODUCIBILITY_FILE_SUFFIX}.json"
            )

    def _initialize_raw_log_file(self):
        """
        Initializes the log file with headers for better readability.
        """
        with open(self.raw_log_filename, "w") as f:
            f.write(
                "Generation|Population|DuplicatedGenes|BestFitness|WorstFitness|MeanFitness|MedianFitness|StdFitness|TimeTaken\n"
            )

    def _write_raw_log(self, message):
        with open(self.raw_log_filename, "a", buffering=8192) as f:
            f.write(message)

    def _load_data(self):
        if self.dataset is None:
            raise ValueError("Dataset not provided")

        if self.dataset == "BCC":
            labeled_train_data = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET, 9)
            labeled_test_data = read_data_from_csv(BREAST_CANCER_TEST_DATASET, 9)

        elif self.dataset == "WR":
            labeled_train_data = read_data_from_csv(WINE_RED_TRAIN_DATASET, 11)
            labeled_test_data = read_data_from_csv(WINE_RED_TRAIN_DATASET, 11)

        else:
            raise ValueError("Invalid dataset")

        self.train_true_labels = [label for label, _ in labeled_train_data]
        self.train_data = [features for _, features in labeled_train_data]

        self.test_true_labels = [label for label, _ in labeled_test_data]
        self.test_data = [features for _, features in labeled_test_data]

        # Add the terminal values to the terminals list based on the number of features
        self.terminals = [
            f"x{i}" for i in range(len(self.train_data[0]))
        ] + TERMINAL_BASE

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

            new_genes = generate_initial_population(
                duplicated_genes, self.terminals, "half_and_half"
            )

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
        terminals,
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
                        terminals,
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
            "TERMINAL": self.terminals,
            "DATASET": self.dataset,
        }

        with open(self.json_filename, "w") as json_file:
            json.dump(data_to_save, json_file, indent=4)

    def run(self, multi_processing_workers=None):
        self._load_data()
        self._initialize_raw_log_file()

        if self.train_data is None or self.train_true_labels is None:
            raise ValueError("Data or true labels not loaded")

        population = generate_initial_population(
            self.population_size, self.terminals, "half_and_half"
        )

        self.handle_duplicate_genes(population)

        self.evaluate_population(population, workers=multi_processing_workers)

        # Evaluate the initial population
        best_genes = self._run_evolution(
            population,
            self.population_size,
            multi_processing_workers=multi_processing_workers,
        )

        self._run_test(best_genes)

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
            start_time = time.time()

            # Elitism: Keep the best ELITISM_SIZE genes from the previous generation
            new_population = heapq.nlargest(self.elitism_size, population)

            # Get the childs and the new seed sequence
            childs, seed_sequence = self.generate_childs(
                population,
                self.terminals,
                seed_sequence,
                self.crossovers_by_generation,
                self.crossover_prob,
                self.mutation_prob,
                workers=multi_processing_workers,
            )

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

        return self.get_unique_best(n_best_fitness, population)

    def _run_test(self, best_genes):
        """
        Run the test phase using the best gene found

        @param best_gene: The best genes found
        """
        self._write_raw_log("-" * 200 + "\n")
        self._write_raw_log(
            "RankingTrain|RankingTest|TrainFitness|TestFitness|GeneTreeHeight\n"
        )

        test_fitness = []

        for i, gene in enumerate(best_genes, start=1):
            test_fitness_value = evaluate_fitness(
                gene, self.test_data, self.test_true_labels
            )
            test_fitness.append((gene, test_fitness_value, i))

        test_fitness.sort(key=lambda x: x[1], reverse=True)

        test_fitness_with_ranks = []

        for rankTest, (gene, test_fitness_value, rankTrain) in enumerate(
            test_fitness, start=1
        ):
            test_fitness_with_ranks.append(
                (gene, test_fitness_value, rankTest, rankTrain)
            )

        test_fitness_with_ranks.sort(key=lambda x: x[3])

        for gene, test_fitness_value, rankTest, rankTrain in test_fitness_with_ranks:
            self._write_raw_log(
                f"{rankTrain}|{rankTest}|{gene.fitness}|{test_fitness_value}|{gene.height}\n"
            )
