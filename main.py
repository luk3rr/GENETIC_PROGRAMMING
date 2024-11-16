#!/usr/bin/env python3

# Filename: main.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import argparse

from gp.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Run the genetic algorithm simulation")

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

    args = parser.parse_args()

    crossovers_by_generation = (
        args.crossovers_by_generation or args.population_size // 2
    )
    elitism_size = args.elitism_size or round(args.population_size * 0.05)

    simulation_id = args.simulation_id or None

    sim = Simulation(
        seed=args.seed,
        thread_initial_seed=args.thread_initial_seed,
        population_size=args.population_size,
        crossovers_by_generation=crossovers_by_generation,
        num_generations=args.num_generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        tournament_size=args.tournament_size,
        elitism_size=elitism_size,
        simulation_id=simulation_id
    )

    sim.run()
    sim.save_simulation_parameters()


if __name__ == "__main__":
    main()
