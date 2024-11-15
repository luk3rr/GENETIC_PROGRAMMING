#!/usr/bin/env python

# Filename: gene.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import numpy as np

from .parameters import DIMENSION

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

    def __hash__(self):
        left_hash = hash(self.left) if self.left else 0
        right_hash = hash(self.right) if self.right else 0
        return hash((self.value, left_hash, right_hash))

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_depth(self):
        return self.depth


class Gene:
    def __init__(self, tree):
        self.root_node = tree
        self.height = self._calculate_tree_height(self.root_node)
        self.fitness: float = 0.0

    def __eq__(self, other):
        """
        Override the equality operator to compare two Gene objects

        @param other: The other Gene object to compare
        @return: True if the two Gene objects are equal, False otherwise
        """
        if not isinstance(other, Gene):
            return False
        return hash(self) == hash(other)

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
        return hash(self.root_node) ^ hash(self.height)

    def calculate_tree_height(self):
        """
        Recalculate and update the height of the entire tree
        """
        self.height = self._calculate_tree_height(self.root_node)

    def _calculate_tree_height(self, node):
        """
        Helper method to calculate the height of a subtree
        """
        if node is None:
            return -1  # Height of an empty tree is -1

        left_height = self._calculate_tree_height(node.left)
        right_height = self._calculate_tree_height(node.right)

        return 1 + max(left_height, right_height)

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

        assert (
            len(ei) == len(ej) == DIMENSION
        ), "The input values must have the same dimension"

        ei = np.array(ei)
        ej = np.array(ej)
        xs = ei - ej

        return self._evaluate_function(self.root_node, xs)

    def _evaluate_function(self, node, values, prefix="x"):
        if node.is_leaf():
            if node.value.startswith(prefix):
                return values[int(node.value[1:])]

            else:
                return int(node.value)

        left = self._evaluate_function(node.left, values, prefix)
        right = self._evaluate_function(node.right, values, prefix)

        if node.value == "+":
            return left + right
        elif node.value == "-":
            return left - right
        elif node.value == "*":
            return left * right
        elif node.value == "/":
            # Protected division
            return left / right if right != 0 else left / 1e-10
        else:
            raise ValueError(f"Invalid operator: {node.value}")

    def show_prefix(self):
        """
        Return the prefix notation of the tree
        """
        return self._show_prefix(self.root_node)

    def _show_prefix(self, node):
        if node is None:
            return " "

        result = str(node.value)

        if node.left is not None:
            result += " " + self._show_prefix(node.left)
        if node.right is not None:
            result += " " + self._show_prefix(node.right)
        return result

    def show_infix(self):
        """
        Return the infix notation of the tree
        """
        return self._show_infix(self.root_node)

    def _show_infix(self, node):
        if node is None:
            return " "

        # Infix notation: Left -> Root -> Right
        if node.is_leaf():
            return str(node.value)

        left_expr = self._show_infix(node.left) if node.left else ""
        right_expr = self._show_infix(node.right) if node.right else ""

        return f"({left_expr} {node.value} {right_expr})"
