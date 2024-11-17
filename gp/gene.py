#!/usr/bin/env python

# Filename: gene.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import numpy as np

from gp.parameters import SimulationConfig

from .utils import calculate_tree_height
from numexpr import evaluate


class Node:
    def __init__(self, value, depth, left=None, right=None):
        """
        Node constructor

        @param value: the value of the node (operator or operand)
        @param depth: the depth of the node in the tree
        @param left: the left child node
        @param right: the right child node
        """
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None


class Gene:
    def __init__(self, tree):
        self.root_node = tree
        self.height = calculate_tree_height(self.root_node)
        self.fitness: float = 0.0
        self.expr = self.get_infix()

    def __eq__(self, other):
        """
        Override the equality operator to compare two Gene objects

        @param other: The other Gene object to compare
        @return: True if the two Gene objects are equal, False otherwise
        """
        if not isinstance(other, Gene):
            return False
        return self.expr == other.expr

    def __ne__(self, other):
        """
        Override the inequality operator to compare two Gene objects

        @param other: The other Gene object to compare
        @return: True if the two Gene objects are not equal, False otherwise
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __hash__(self):
        return hash(self.expr)

    def build_infix_expr(self):
        """
        Build the infix expression of the gene
        """
        self.expr = self.get_infix()

    def calculate_tree_height(self):
        """
        Recalculate and update the height of the entire tree
        """
        self.height = calculate_tree_height(self.root_node)

    def get_distance_matrix(self, data) -> np.ndarray:
        """
        Create a distance matrix for the data using the gene's evaluate function

        @param data: The data to make the distance matrix, as a list of tuples (label, features)
        @return: The distance matrix as a NumPy array
        """
        num_data_points = len(data)
        distance_matrix = np.zeros((num_data_points, num_data_points))

        for x in range(num_data_points):
            features_x = data[x]

            for y in range(x + 1, num_data_points):
                features_y = data[y]

                distance = self.evaluate_function(features_x, features_y)

                # Matrix is symmetric, so we can fill both values at once
                distance_matrix[x, y] = distance
                distance_matrix[y, x] = distance

        return distance_matrix

    def evaluate_function(self, ei, ej):
        """
        Evaluate the gene using the input values

        @param ei: The first input values
        @param ej: The second input values
        """
        ei = np.array(ei)
        ej = np.array(ej)
        xs = ei - ej

        return self._evaluate_function(xs)

    def _evaluate_function(self, values, prefix="x"):
        """
        Evaluate the gene using the input values

        @param values: The input values
        @param prefix: The prefix to use in the expression
        """
        config = SimulationConfig().get_args()

        terminals_list = config.terminals

        terminals_dict = {}

        for i, value in enumerate(values):
            key = f"{prefix}{i}"
            if key in terminals_list:
                terminals_dict[key] = value

        return evaluate(self.expr, terminals_dict)

    def get_prefix(self):
        """
        Return the prefix notation of the tree
        """
        return self._get_prefix(self.root_node)

    def _get_prefix(self, node):
        if node is None:
            return " "

        result = str(node.value)

        if node.left is not None:
            result += " " + self._get_prefix(node.left)
        if node.right is not None:
            result += " " + self._get_prefix(node.right)
        return result

    def get_infix(self):
        """
        Return the infix notation of the tree
        """
        return self._get_infix(self.root_node)

    def _get_infix(self, node) -> str:
        """
        Return the infix notation of the tree with modifications for division and subtraction.

        @param node: The node to start the infix notation
        @return: The infix notation of the tree
        """
        if node is None:
            return " "

        if node.is_leaf():
            return str(node.value)

        left_expr = self._get_infix(node.left) if node.left else ""
        right_expr = self._get_infix(node.right) if node.right else ""

        if node.value == "/":  # Protected division
            return f"(abs({left_expr} / ({right_expr} + 1e-10)))"
        elif node.value == "-":  # Absolute difference
            return f"(abs({left_expr} - {right_expr}))"
        else:
            return f"({left_expr} {node.value} {right_expr})"
