#!/usr/bin/env python3

# Filename: __main__.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import os

from .summarizer import process_experiment_logs
from .analyzer import *
from .constants import LOG_FOLDER, FIGS_FOLDER, print_line, print_separator


def main():
    os.makedirs(FIGS_FOLDER, exist_ok=True)

    # Process the experiment logs
    process_experiment_logs(LOG_FOLDER)

    # Load the data
    training_df, ranking_df = load_data()

    merged_df = pd.merge(training_df, ranking_df, on="ExperimentId")

    print_line()

    params = {
        "Generations": 80,
        "CrossoverRate": 0.9,
        "MutationRate": 0.05,
        "TournamentSize": 3,
        "ElitismEnabled": 1,
    }

    print("Analyzing the population effect...")
    print("Params: ", params)
    df_population_effect = analyze_population_effect(merged_df, params)

    print(df_population_effect)

    print_line()

    params = {
        "PopulationSize": 100,
        "CrossoverRate": 0.9,
        "MutationRate": 0.05,
        "TournamentSize": 3,
        "ElitismEnabled": 1,
    }

    print("Analyzing the generations effect...")
    print("Params: ", params)

    df_generations_effect = analyze_generation_effect(merged_df, params)
    print(df_generations_effect)

    params["PopulationSize"] = 200

    print_separator()

    print("Params: ", params)
    df_generations_effect = analyze_generation_effect(merged_df, params)
    print(df_generations_effect)

    print_line()

    print("Analyzing the mutation rate effect...")
    params = {
        "PopulationSize": 200,
        "CrossoverRate": 0.9,
        "Generations": 20,
        "TournamentSize": 3,
        "ElitismEnabled": 1,
    }

    print("Params: ", params)
    df_mutation_effect = analyze_mutation_effect(merged_df, params)
    print(df_mutation_effect)

    print_separator()

    params["Generations"] = 80
    print("Params: ", params)
    df_mutation_effect = analyze_mutation_effect(merged_df, params)
    print(df_mutation_effect)

    print_line()

    print("Analyzing the tournament size effect...")
    params = {
        "PopulationSize": 200,
        "CrossoverRate": 0.9,
        "Generations": 80,
        "MutationRate": 0.05,
        "ElitismEnabled": 1,
    }

    print("Params: ", params)
    df_tournament_effect = analyze_tournament_size_effect(merged_df, params)
    print(df_tournament_effect)

    print_line()

    plot_top_parameter_sets(merged_df)

    print_line()

    best_parameter_set = get_extreme_fitness(merged_df)

    print("Best parameter set: \n", best_parameter_set)

if __name__ == "__main__":
    main()
