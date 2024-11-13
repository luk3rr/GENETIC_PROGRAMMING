#!/usr/bin/env python3

# Filename: utils.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from colorama import Fore, Style

from .parameters import NON_TERMINAL


def print_line():
    print("-" * 80)


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
