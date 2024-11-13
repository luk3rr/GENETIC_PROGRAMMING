#!/usr/bin/env python

# Filename: gene.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from .utils import are_trees_equal

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

    def get_depth(self):
        return self.depth

class Gene:
    def __init__(self, tree):
        self.root_node = tree
        self.height = self._calculate_height(self.root_node)

    def __eq__(self, other):
        """
        Override the equality operator to compare two Gene objects

        @param other: The other Gene object to compare
        @return: True if the two Gene objects are equal, False otherwise
        """
        if not isinstance(other, Gene):
            return False
        return are_trees_equal(self.root_node, other.root_node)

    def __ne__(self, other):
        """
        Override the inequality operator to compare two Gene objects

        @param other: The other Gene object to compare
        @return: True if the two Gene objects are not equal, False otherwise
        """
        return not self.__eq__(other)

    def recalculate_height(self):
        """
        Recalculate and update the height of the entire tree
        """
        self.height = self._calculate_height(self.root_node)

    def _calculate_height(self, node):
        """
        Helper method to calculate the height of a subtree
        """
        if node is None:
            return -1  # Considera altura -1 para subárvores vazias

        left_height = self._calculate_height(node.left)
        right_height = self._calculate_height(node.right)

        return 1 + max(left_height, right_height)

    def evaluate(self, ei, ej):
        """"""
        pass

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
