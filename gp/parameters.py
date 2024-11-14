#!/usr/bin/env python

# Filename: parameters.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

SEED = 42

TREE_MAX_DEPTH = 7
DIMENSION = 9

NON_TERMINAL_PROB = 0.5
TERMINAL_PROB = 1 - NON_TERMINAL_PROB

POPULATION_SIZE = 200
NUM_GENERATIONS = 50

CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.05

TOURNAMENT_SIZE = 5
ELITISM_SIZE = round(POPULATION_SIZE * 0.1)

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
