#!/usr/bin/env python

# Filename: parameters.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

BREAST_CANCER_TEST_DATASET = "data/breast_cancer_coimbra_test.csv"
BREAST_CANCER_TRAIN_DATASET = "data/breast_cancer_coimbra_train.csv"

SEED = 42

TREE_MAX_DEPTH = 7
DIMENSION = 9

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

NON_TERMINAL_PROB = 0.5
TERMINAL_PROB = 1 - NON_TERMINAL_PROB

POPULATION_SIZE = 1000
NUM_GENERATIONS = 500

CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.05

TOURNAMENT_SIZE = 5
ELITISM_SIZE = round(POPULATION_SIZE * 0.1)
