#!/usr/bin/env python

# Filename: parameters.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

SEED = 2427935994
THREAD_INITIAL_SEED = 3234817

TREE_MAX_DEPTH = 7
TREE_MIN_DEPTH = 1

assert TREE_MAX_DEPTH >= TREE_MIN_DEPTH and TREE_MIN_DEPTH > 0, "Invalid tree depth"

DIMENSION = 9

NON_TERMINAL_PROB = 0.5
TERMINAL_PROB = 1 - NON_TERMINAL_PROB

POPULATION_SIZE = 250
CROSSOVERS_BY_GENERATION = POPULATION_SIZE // 2
NUM_GENERATIONS = 50

CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.1

TOURNAMENT_SIZE = 2
ELITISM_SIZE = round(POPULATION_SIZE * 0.05)

NON_TERMINAL = ["+", "-", "*", "/"]
TERMINAL = [
    "x0",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
]

DATA_FOLDER = "data/"
LOG_FOLDER = "log/"

BREAST_CANCER_TEST_DATASET = DATA_FOLDER + "breast_cancer_coimbra_test.csv"
BREAST_CANCER_TRAIN_DATASET = DATA_FOLDER + "breast_cancer_coimbra_train.csv"

LOG_PREFIX = "sim"
