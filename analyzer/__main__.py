#!/usr/bin/env python3

# Filename: __main__.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from .summarizer import process_experiment_logs
from .constants import LOG_FOLDER


def main():
    process_experiment_logs(LOG_FOLDER)


if __name__ == "__main__":
    main()
