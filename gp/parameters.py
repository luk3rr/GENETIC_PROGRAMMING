#!/usr/bin/env python

# Filename: parameters.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from argparse import Namespace, ArgumentParser
from enum import Enum

TREE_MAX_DEPTH = 7
TREE_MIN_DEPTH = 3

assert TREE_MAX_DEPTH >= TREE_MIN_DEPTH and TREE_MIN_DEPTH > 0, "Invalid tree depth"

NON_TERMINAL_PROB = 0.5
TERMINAL_PROB = 1 - NON_TERMINAL_PROB

NON_TERMINAL = ["+", "-", "*", "/"]

# The base of the terminal set is the range of the data dimension
TERMINAL_BASE = [
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


class NormalizationMethod(Enum):
    MIN_MAX = "MIN_MAX"
    STANDARD = "Z_SCORE"

    def __str__(self):
        return self.value


NORMALIZATION_METHOD = NormalizationMethod.STANDARD


class SimulationConfig:
    """
    Singleton class to store the simulation parameters
    """

    _instance = None

    @classmethod
    def get_args(self) -> Namespace:
        if self._instance is None:
            self._instance = parser.parse_args()

        return self._instance


parser = ArgumentParser(description="Run the genetic algorithm simulation")

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=982623834,
    help="Seed for random number generation",
)
parser.add_argument(
    "-ts",
    "--thread-initial-seed",
    type=int,
    default=433812312,
    help="Initial seed for threads",
)
parser.add_argument(
    "-p", "--population-size", type=int, default=300, help="Size of the population"
)
parser.add_argument(
    "-c",
    "--crossovers-by-generation",
    type=int,
    help="Number of crossovers per generation. Default is half of the population size",
)
parser.add_argument(
    "-g", "--num-generations", type=int, default=100, help="Number of generations"
)
parser.add_argument(
    "-cp",
    "--crossover-prob",
    type=float,
    default=0.9,
    help="Probability of crossover",
)
parser.add_argument(
    "-mp",
    "--mutation-prob",
    type=float,
    default=0.1,
    help="Probability of mutation",
)
parser.add_argument(
    "-t", "--tournament-size", type=int, default=2, help="Tournament size"
)
parser.add_argument(
    "-e",
    "--elitism-size",
    type=float,
    help="Elitism size as a fraction of population size. Default is 5%% of the population size",
)
parser.add_argument(
    "-i",
    "--simulation-id",
    type=str,
    help="Simulation identifier. Used to save the simulation data. Default is the current timestamp",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    help="Dataset to use. Options are 'BCC' and 'WR'",
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    help="Number of workers to use for parallel processing. If not specified, only main process is used",
)
