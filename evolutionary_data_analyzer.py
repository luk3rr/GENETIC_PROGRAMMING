#!/usr/bin/env python3

# Filename: evolutionary_data_analyzer.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from io import StringIO

LOG_FOLDER = "testlog/"
OUTPUT_FOLDER = "processed/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_PREFIX_LENGTH = 2
EXPERIMENT_ROUNDS = 10

TRAINING_SUMMARY_CSV = "training_summary.csv"
RANKING_SUMMARY_CSV = "ranking_summary.csv"

# Mapping between the identifier prefix and the identifier name
IDENTIFIER_NAMES = {
    "SD": "seed",
    "PS": "population_size",
    "GS": "generations",
    "PC": "crossover_rate",
    "PM": "mutation_rate",
    "TS": "tournament_size",
    "EE": "elitism_enabled",
}

# All data are mean values
RANKING_SUMMARY_COLUMNS = [
    "ExperimentId",
    "BestFitnessOnTest", # Best fitness on test data
    "MeanFitnessOnTest", # Mean fitness on test data
    "StdFitnessOnTest",  # Standard deviation of fitness on test data
    "BestFitnessOnTraining", # Best fitness on training data
    "MeanFitnessOnTraining", # Mean fitness on training data
    "StdFitnessOnTraining",  # Standard deviation of fitness on training data
    "BestGeneTestPositionOnTraining", # Best gene on test data position on training data
    "BestGeneTrainingPositionOnTest", # Best gene on training data position on test data
    "BestGeneTestHeight", # Best gene on test data height
    "BestGeneTrainingHeight", # Best gene on training data height
    "RMSE", # Root Mean Squared Error between test and training data
    "SpearmansCorrelation", # Spearmans correlation between ranking of test and training data
]

# All data are mean values
TRAINING_SUMMARY_COLUMNS = [
    "ExperimentId",
    "DuplicatedGenes", # Number of duplicated genes in each generation
    "GeneratedChilds", # Number of generated childs in each generation
    "BetterChilds", # Number of childs with fitness better than the mean fitness of the parents in each generation
    "WorstChilds", # Number of childs with fitness worse than the mean fitness of the parents in each generation
    "BestFitness", # Best fitness in each generation
    "WorstFitness", # Worst fitness in each generation
    "MeanFitness", # Mean fitness in each generation
    "MedianFitness", # Median fitness in each generation
    "StdFitness", # Standard deviation of fitness in each generation
    "TimeTaken", # Time taken to process each generation
]



def training_summary(df_generation) -> dict:
    """ """
    # Mensure the mean of all columns and save it
    mean_values = pd.concat(df_generation).mean()

    if isinstance(mean_values, pd.Series):
        return mean_values.to_dict()
    else:
        return {"mean": mean_values}


def ranking_summary(df_ranking) -> dict:
    """
    Summarize the ranking data

    @param df_ranking: the DataFrame with the ranking data
    @return: a dictionary with the summary data
    """
    best_fitness_on_test = df_ranking["TestFitness"].max()
    mean_fitness_on_test = df_ranking["TestFitness"].mean()
    std_fitness_on_test = df_ranking["TestFitness"].std()

    best_fitness_on_training = df_ranking["TrainFitness"].max()
    mean_fitness_on_training = df_ranking["TrainFitness"].mean()
    std_fitness_on_training = df_ranking["TrainFitness"].std()

    # Get the best gene data. Handle the case of ties or duplicated genes
    best_gene_test_position_on_training = df_ranking.loc[
        df_ranking["TrainFitness"].idxmax(), "RankingTest"
    ]

    best_gene_test_position_on_training = (
        df_ranking.loc[df_ranking["TrainFitness"] == df_ranking["TrainFitness"].max()]
        .sort_values("RankingTest")
        .iloc[0]["RankingTest"]
    )

    best_gene_training_position_on_test = df_ranking.loc[
        df_ranking["TestFitness"].idxmax(), "RankingTrain"
    ]

    best_gene_training_position_on_test = (
        df_ranking.loc[df_ranking["TestFitness"] == df_ranking["TestFitness"].max()]
        .sort_values("RankingTrain")
        .iloc[0]["RankingTrain"]
    )

    best_gene_test_height = df_ranking.loc[
        df_ranking["TestFitness"].idxmax(), "GeneTreeHeight"
    ]
    best_gene_test_height = (
        df_ranking.loc[df_ranking["TestFitness"] == df_ranking["TestFitness"].max()]
        .sort_values("GeneTreeHeight")
        .iloc[0]["GeneTreeHeight"]
    )

    best_gene_training_height = df_ranking.loc[
        df_ranking["TrainFitness"].idxmax(), "GeneTreeHeight"
    ]
    best_gene_training_height = (
        df_ranking.loc[df_ranking["TrainFitness"] == df_ranking["TrainFitness"].max()]
        .sort_values("GeneTreeHeight")
        .iloc[0]["GeneTreeHeight"]
    )

    # Root Mean Squared Error
    rmse = np.sqrt(
        ((df_ranking["TestFitness"] - df_ranking["TrainFitness"]) ** 2).mean()
    )

    # Spearmans Correlation
    corr, pvalue = spearmanr(df_ranking["TrainFitness"], df_ranking["TestFitness"])

    return {
        "BestFitnessOnTest": best_fitness_on_test,
        "MeanFitnessOnTest": mean_fitness_on_test,
        "StdFitnessOnTest": std_fitness_on_test,
        "BestFitnessOnTraining": best_fitness_on_training,
        "MeanFitnessOnTraining": mean_fitness_on_training,
        "StdFitnessOnTraining": std_fitness_on_training,
        "BestGeneTestPositionOnTraining": best_gene_test_position_on_training,
        "BestGeneTrainingPositionOnTest": best_gene_training_position_on_test,
        "BestGeneTestHeight": best_gene_test_height,
        "BestGeneTrainingHeight": best_gene_training_height,
        "RMSE": rmse,
        "SpearmansCorrelation": corr,
        "SpearmansCorrPValue": pvalue,
    }


def parse_experiment_id(filename) -> dict:
    """
    Extracts the identifiers from the experiment filename

    @param filename: the filename of the experiment
    @return: a dictionary with the identifiers
    """
    parts = filename.split("_")
    experiment_data = {}

    for part in parts:
        prefix = part[:ID_PREFIX_LENGTH]

        if prefix in IDENTIFIER_NAMES:
            key = IDENTIFIER_NAMES[prefix]
            value = part[ID_PREFIX_LENGTH:]
            experiment_data[key] = value

    return experiment_data


def get_experiment_id_from_file(file_path):
    """
    Retorna o identificador do experimento a partir do caminho do arquivo
    """
    file_name = os.path.basename(file_path)

    start = file_name.find("PS")
    end = file_name.find("_SALT")

    return file_name[start:end]


def process_experiment_file(file_path) -> tuple:
    """
    Process the experiment file and return the DataFrames with the data

    @param file_path: the path to the experiment file
    @return: a tuple with the DataFrames for the generations and rankings
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the separator between generations and rankings
    separator_index = next(
        (i for i, line in enumerate(lines) if line.startswith("-")), len(lines)
    )

    generation_data = lines[:separator_index]
    ranking_data = lines[separator_index + 1 :]

    df_generation = pd.read_csv(StringIO("".join(generation_data)), sep="|")
    df_ranking = pd.read_csv(StringIO("".join(ranking_data)), sep="|")

    return df_generation, df_ranking


def get_all_experiments(log_folder) -> dict:
    """
    Get all the experiments in the log folder

    @param log_folder: the folder with the log files
    @return: a dictionary with the experiment ids and the files
    """
    experiment_ids = {}

    for file_name in os.listdir(log_folder):
        if file_name.endswith(".dat"):
            experiment_id = get_experiment_id_from_file(file_name)

            if experiment_id not in experiment_ids:
                experiment_ids[experiment_id] = []

            experiment_ids[experiment_id].append(file_name)

    # for experiment in experiment_ids:
    #    assert (
    #        len(experiment_ids[experiment]) == EXPERIMENT_ROUNDS
    #    ), f"Experiment {experiment} has {len(experiment_ids[experiment])} rounds"

    return experiment_ids


def process_experiment_logs(
    log_folder, output_folder=OUTPUT_FOLDER, output_file=TRAINING_SUMMARY_CSV
):
    """
    Processa todos os arquivos na pasta de logs e salva dados consolidados.
    """
    experiments = get_all_experiments(log_folder)
    training_summary_data = []
    ranking_summary_data = []

    for experiment_id, files in experiments.items():
        generation_dfs = []
        ranking_dfs = []

        for file in files:
            file_path = os.path.join(log_folder, file)
            df_generation, df_ranking = process_experiment_file(file_path)

            generation_dfs.append(df_generation)
            ranking_dfs.append(df_ranking)

        # Remove the generation number column and population size
        generation_dfs = [
            df.drop(columns=["Generation", "Population"]) for df in generation_dfs
        ]

        # Process the training data
        training_summary_dict = training_summary(generation_dfs)
        training_summary_dict["ExperimentId"] = experiment_id

        training_summary_data.append(training_summary_dict)

        # Process the ranking data
        ranking_summary_dict = ranking_summary(pd.concat(ranking_dfs))
        ranking_summary_dict["ExperimentId"] = experiment_id

        ranking_summary_data.append(ranking_summary_dict)

    # Save the training summary data
    training_summary_df = pd.DataFrame(training_summary_data)
    training_summary_df = training_summary_df[
        ["ExperimentId"]
        + [col for col in training_summary_df.columns if col != "ExperimentId"]
    ]

    training_summary_df.to_csv(
        os.path.join(output_folder, output_file), index=False, sep="|"
    )

    # Save the ranking summary data
    ranking_summary_df = pd.DataFrame(ranking_summary_data)
    ranking_summary_df = ranking_summary_df[
        ["ExperimentId"]
        + [col for col in ranking_summary_df.columns if col != "ExperimentId"]
    ]

    ranking_summary_df.to_csv(
        os.path.join(output_folder, RANKING_SUMMARY_CSV), index=False, sep="|"
    )


def main():
    process_experiment_logs(LOG_FOLDER)


if __name__ == "__main__":
    main()