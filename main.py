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
    sim = Simulation()

    sim.run()

if __name__ == "__main__":
    main()
