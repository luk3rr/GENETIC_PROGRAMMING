#!/usr/bin/env python3

# Filename: main.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from gp.simulation import Simulation
from gp.gene import *
from gp.population import *
from gp.parameters import *

from gp.test import *


def main():
    seed = 982623834
    thread_initial_seed = 433812312

    sim = Simulation(seed=seed, thread_initial_seed=thread_initial_seed)

    sim.run()
    sim.save_simulation_parameters()


if __name__ == "__main__":
    main()
