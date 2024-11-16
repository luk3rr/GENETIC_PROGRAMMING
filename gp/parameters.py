#!/usr/bin/env python

# Filename: parameters.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

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
