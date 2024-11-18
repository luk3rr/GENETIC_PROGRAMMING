#!/usr/bin/env python3

# Filename: __main__.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

from .summarizer import process_experiment_logs
from .analyzer import *
from .constants import LOG_FOLDER, FIGS_FOLDER, print_line


def main():
    os.makedirs(FIGS_FOLDER, exist_ok=True)

    # Process the experiment logs
    process_experiment_logs(LOG_FOLDER)

    # Load the data
    training_df, ranking_df = load_data()

    merged_df = pd.merge(training_df, ranking_df, on="ExperimentId")

    print_line()

    plot_correlation_between_training_and_test(merged_df)

    print_line()

    plot_fitness_vs_mutation_rate(merged_df)

    print_line()

    plot_fitness_vs_crossover_rate(merged_df)

    print_line()

    plot_fitness_vs_population_size(merged_df)

    print_line()

    plot_fitness_vs_generations(merged_df)

    print_line()

    plot_fitness_vs_tournament_size(merged_df)

    print_line()

    plot_fitness_vs_elitism_enabled(merged_df)

    print_line()

    print("Parameter combinations with less ranking discrepancy...")
    print(get_parameter_combinations_with_less_ranking_discrepancy(merged_df))

    print_separator()

    print("Parameter combinations with more ranking discrepancy...")
    print(get_parameter_combinations_with_more_ranking_discrepancy(merged_df))

    print_line()

    print("Parameter combinations with best fitness on test data")
    print(get_parameter_combinations_with_best_fitness_on_test(merged_df))

    print_separator()

    print("Parameter combinations with worst fitness on test data")
    print(get_parameter_combinations_with_worst_fitness_on_test(merged_df))



if __name__ == "__main__":
    main()
