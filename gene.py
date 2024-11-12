#!/usr/bin/env python

# Filename: gene.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

class Gene:
    def __init__(self, tree):
        self.root_node = tree
        self.fitness = None

    def evaluate(self, ei, ej):
        """"""
        pass
