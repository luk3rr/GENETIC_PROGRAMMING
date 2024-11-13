#!/usr/bin/env python3

# Filename: main.py
# Created on: November 12, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from gp.gene import *
from gp.population import *
from gp.parameters import *

from gp.test import *


def main():
    bct_test = read_data_from_csv(BREAST_CANCER_TEST_DATASET)

    bct_train = read_data_from_csv(BREAST_CANCER_TRAIN_DATASET)

    # print all the data
    print("Test data:")
    print(bct_test)

    print("\nTrain data:")
    print(bct_train)

if __name__ == "__main__":
    main()
