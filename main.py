#!/usr/bin/env python3

# Filename: main.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from multiprocessing import cpu_count
from gp.simulation import Simulation
from gp.parameters import SimulationConfig


def main():
    # Get the simulation parameters by singleton
    args = SimulationConfig().get_args()

    # Define derived parameters
    args.crossovers_by_generation = (
        args.crossovers_by_generation or args.population_size // 2
    )
    args.elitism_size = args.elitism_size or round(args.population_size * 0.05)

    args.simulation_id = args.simulation_id or None

    assert (
        args.workers is None or args.workers <= cpu_count() and args.workers > 0
    ), "The number of workers must be between 1 and the number of CPUs available"

    sim = Simulation()
    sim.run()
    sim.save_simulation_parameters()


if __name__ == "__main__":
    main()
