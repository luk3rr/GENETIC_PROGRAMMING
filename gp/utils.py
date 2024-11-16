#!/usr/bin/env python3

# Filename: utils.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import csv
from typing import Tuple

from colorama import Fore, Style
from typing import List, Tuple

from .parameters import DATA_DIMENSION, NON_TERMINAL


def print_line():
    print("-" * 80)


def read_data_from_csv(file_path) -> List[Tuple[int, list]]:
    """
    Reads the data from a CSV file

    Each tuple has the label and the list of features
    NOTE: The last column of the CSV file must be the label

    @param file_path: The path to the CSV file
    @return: A list with the data
    """
    data = []

    with open(file_path, mode="r") as csvfile:
        csv_reader = csv.reader(csvfile)

        # Ignore the header
        next(csv_reader)

        for row in csv_reader:
            values = [float(value) for value in row[:DATA_DIMENSION]]
            label = int(row[DATA_DIMENSION])
            data.append((label, values))

    return data


def are_trees_equal(node1, node2):
    """
    Compares two trees to check if they are equal

    @param node1: The root node of the first tree
    @param node2: The root node of the second tree
    @return: True if the trees are equal, False otherwise
    """
    # Base case: if both nodes are None, the trees are equal
    if node1 is None and node2 is None:
        return True

    # If one of the nodes is None, the trees are different
    if node1 is None or node2 is None:
        return False

    # If the values of the nodes are different, the trees are different
    if node1.value != node2.value:
        return False

    # Recursively check the left and right subtrees
    return are_trees_equal(node1.left, node2.left) and are_trees_equal(
        node1.right, node2.right
    )


def update_nodes_depth(node, current_depth=0):
    """ """
    if node is None:
        return

    # Define a profundidade do nó atual
    node.depth = current_depth

    # Calcula a altura das subárvores esquerda e direita
    update_nodes_depth(node.left, current_depth + 1)
    update_nodes_depth(node.right, current_depth + 1)


def calculate_tree_height(node):
    """
    Helper method to calculate the height of a subtree

    @param node: The root node of the subtree
    """
    if node is None:
        return -1

    left_height = calculate_tree_height(node.left)
    right_height = calculate_tree_height(node.right)

    return 1 + max(left_height, right_height)


def print_tree(node, indent="", last="updown"):
    """
    Print the tree in a readable format

    @param node: The root node of the tree
    @param indent: The current indentation
    @param last: The last direction of the tree
    """
    if node.right:
        next_last = "up" if node.left else "updown"
        print_tree(
            node.right, indent + ("     " if last == "up" else "|    "), next_last
        )

    color = Fore.RED if node.value in NON_TERMINAL else Fore.GREEN

    print(
        indent
        + ("|-- " if last == "updown" else ("\\-- " if last == "up" else "/-- "))
        + color
        + str(node.value)
        + Style.RESET_ALL
    )

    if node.left:
        next_last = "down" if node.right else "updown"
        print_tree(
            node.left, indent + ("     " if last == "down" else "|    "), next_last
        )
