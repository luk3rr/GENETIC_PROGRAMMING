#!/usr/bin/env python3

# Filename: constants.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

LOG_FOLDER = "testlog/"
OUTPUT_FOLDER = "processed/"

ID_PREFIX_LENGTH = 2
EXPERIMENT_ROUNDS = 10

TRAINING_SUMMARY_CSV = "training_summary.csv"
RANKING_SUMMARY_CSV = "ranking_summary.csv"

# Mapping between the identifier prefix and the identifier name
IDENTIFIER_NAMES = {
    "SD": "seed",
    "PS": "population_size",
    "GS": "generations",
    "PC": "crossover_rate",
    "PM": "mutation_rate",
    "TS": "tournament_size",
    "EE": "elitism_enabled",
}

# All data are mean values
RANKING_SUMMARY_COLUMNS = [
    "ExperimentId",
    "BestFitnessOnTest", # Best fitness on test data
    "MeanFitnessOnTest", # Mean fitness on test data
    "StdFitnessOnTest",  # Standard deviation of fitness on test data
    "BestFitnessOnTraining", # Best fitness on training data
    "MeanFitnessOnTraining", # Mean fitness on training data
    "StdFitnessOnTraining",  # Standard deviation of fitness on training data
    "BestGeneTestPositionOnTraining", # Best gene on test data position on training data
    "BestGeneTrainingPositionOnTest", # Best gene on training data position on test data
    "BestGeneTestHeight", # Best gene on test data height
    "BestGeneTrainingHeight", # Best gene on training data height
    "RMSE", # Root Mean Squared Error between test and training data
    "SpearmansCorrelation", # Spearmans correlation between ranking of test and training data
]

# All data are mean values
TRAINING_SUMMARY_COLUMNS = [
    "ExperimentId",
    "DuplicatedGenes", # Number of duplicated genes in each generation
    "GeneratedChilds", # Number of generated childs in each generation
    "BetterChilds", # Number of childs with fitness better than the mean fitness of the parents in each generation
    "WorstChilds", # Number of childs with fitness worse than the mean fitness of the parents in each generation
    "BestFitness", # Best fitness in each generation
    "WorstFitness", # Worst fitness in each generation
    "MeanFitness", # Mean fitness in each generation
    "MedianFitness", # Median fitness in each generation
    "StdFitness", # Standard deviation of fitness in each generation
    "TimeTaken", # Time taken to process each generation
]
