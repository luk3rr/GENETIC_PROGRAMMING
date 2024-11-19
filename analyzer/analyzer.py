#!/usr/bin/env python3

# Filename: analyzer.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

"""
This script is responsible for analyzing the data summarized by the summarizer.py script.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os

from typing import Tuple

from .constants import OUTPUT_FOLDER, TRAINING_SUMMARY_CSV, RANKING_SUMMARY_CSV, FIGS_FOLDER, FIGS_FORMAT, FIGS_DPI


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the summarized data from the summarizer.py script

    @return: A tuple containing the training data and the ranking data
    """
    training_data = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, TRAINING_SUMMARY_CSV), sep="|"
    )
    ranking_data = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, RANKING_SUMMARY_CSV), sep="|"
    )

    return training_data, ranking_data


def analyze_population_effect(df, params) -> pd.DataFrame:
    """
    Analyzes the impact of population size on the quality of individuals, keeping the other parameters fixed

    @param df: DataFrame containing the combined data for analysis
    @param params: Dictionary containing the values of the fixed parameters
                   (ex.: {'Generations': 20, 'PopulationSize': 100, ...})
    """
    filtered_df = df

    for key, value in params.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        raise ValueError("Nenhum dado encontrado com os parâmetros fornecidos.")

    sorted_df = filtered_df.sort_values("PopulationSize")

    results = []
    for i in range(len(sorted_df) - 1):
        row1 = sorted_df.iloc[i]
        row2 = sorted_df.iloc[i + 1]

        diffTraining = row2["MeanFitnessOnTraining"] - row1["MeanFitnessOnTraining"]
        improvementTraining = (diffTraining / row1["MeanFitnessOnTraining"]) * 100

        diffTest = row2["MeanFitnessOnTest"] - row1["MeanFitnessOnTest"]
        improvementTest = (diffTest / row1["MeanFitnessOnTest"]) * 100

        results.append(
            {
                "Population1": row1["PopulationSize"],
                "Population2": row2["PopulationSize"],
                "FitnessTraining1": row1["MeanFitnessOnTraining"],
                "FitnessTraining2": row2["MeanFitnessOnTraining"],
                "FitnessTest1": row1["MeanFitnessOnTest"],
                "FitnessTest2": row2["MeanFitnessOnTest"],
                # "AbsoluteImprovementTraining": diffTraining,
                "PercentImprovementTraining": improvementTraining,
                # "AbsoluteImprovementTest": diffTest,
                "PercentImprovementTest": improvementTest,
            }
        )

    return pd.DataFrame(results)


def analyze_generation_effect(df, params) -> pd.DataFrame:
    """
    Analyzes the impact of the number of generations on the quality of individuals,
    keeping the other parameters fixed

    @param df: DataFrame containing the combined data for analysis
    @param params: Dictionary containing the values of the fixed parameters
                   (ex.: {'Generations': 20, 'PopulationSize
    """
    filtered_df = df
    for key, value in params.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        raise ValueError("Nenhum dado encontrado com os parâmetros fornecidos.")

    sorted_df = filtered_df.sort_values("Generations")

    results = []
    for i in range(len(sorted_df) - 1):
        row1 = sorted_df.iloc[i]
        row2 = sorted_df.iloc[i + 1]

        diffTraining = row2["MeanFitnessOnTraining"] - row1["MeanFitnessOnTraining"]
        improvementTraining = (diffTraining / row1["MeanFitnessOnTraining"]) * 100

        diffTest = row2["MeanFitnessOnTest"] - row1["MeanFitnessOnTest"]
        improvementTest = (diffTest / row1["MeanFitnessOnTest"]) * 100

        results.append(
            {
                "Generations1": row1["Generations"],
                "Generations2": row2["Generations"],
                "FitnessTraining1": row1["MeanFitnessOnTraining"],
                "FitnessTraining2": row2["MeanFitnessOnTraining"],
                "FitnessTest1": row1["MeanFitnessOnTest"],
                "FitnessTest2": row2["MeanFitnessOnTest"],
                # "AbsoluteImprovementTraining": diffTraining,
                "PercentImprovementTraining": improvementTraining,
                # "AbsoluteImprovementTest": diffTest,
                "PercentImprovementTest": improvementTest,
            }
        )

    return pd.DataFrame(results)


def analyze_mutation_effect(df, params) -> pd.DataFrame:
    """
    Analyzes the impact of the mutation rate on the quality of individuals, keeping the other parameters fixed

    @param df: DataFrame containing the combined data for analysis
    @param params: Dictionary containing the values of the fixed parameters
                   (ex.: {'Generations': 20, 'PopulationSize': 100, ...})
    """
    filtered_df = df
    for key, value in params.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        raise ValueError("Nenhum dado encontrado com os parâmetros fornecidos.")

    sorted_df = filtered_df.sort_values("MutationRate")

    results = []
    for i in range(len(sorted_df) - 1):
        row1 = sorted_df.iloc[i]
        row2 = sorted_df.iloc[i + 1]

        better_diff = row2["BetterChilds"] - row1["BetterChilds"]
        worse_diff = row2["WorstChilds"] - row1["WorstChilds"]

        improvementBetter = (better_diff / row1["BetterChilds"]) * 100

        improvementWorse = (worse_diff / row1["WorstChilds"]) * 100

        results.append(
            {
                "MutationRate1": row1["MutationRate"],
                "MutationRate2": row2["MutationRate"],
                "BetterChilds1": row1["BetterChilds"],
                "BetterChilds2": row2["BetterChilds"],
                "BetterChildsDiff": better_diff,
                "WorstChilds1": row1["WorstChilds"],
                "WorstChilds2": row2["WorstChilds"],
                "WorstChildsDiff": worse_diff,
                "PercentImprovementBetter": improvementBetter,
                "PercentImprovementWorse": improvementWorse,
            }
        )

    return pd.DataFrame(results)

def analyze_tournament_size_effect(df, params) -> pd.DataFrame:
    """
    Analyzes the impact of the tournament size on the quality of individuals, keeping the other parameters fixed.

    @param df: DataFrame containing the combined data for analysis.
    @param params: Dictionary containing the values of the fixed parameters.
    """
    filtered_df = df
    for key, value in params.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        raise ValueError("Nenhum dado encontrado com os parâmetros fornecidos.")

    sorted_df = filtered_df.sort_values("TournamentSize")

    results = []
    for i in range(len(sorted_df) - 1):
        row1 = sorted_df.iloc[i]
        row2 = sorted_df.iloc[i + 1]

        better_diff = row2["BetterChilds"] - row1["BetterChilds"]
        worse_diff = row2["WorstChilds"] - row1["WorstChilds"]

        improvementBetter = (better_diff / row1["BetterChilds"]) * 100
        improvementWorse = (worse_diff / row1["WorstChilds"]) * 100

        results.append(
            {
                "TournamentSize1": row1["TournamentSize"],
                "TournamentSize2": row2["TournamentSize"],
                "BetterChilds1": row1["BetterChilds"],
                "BetterChilds2": row2["BetterChilds"],
                "BetterChildsDiff": better_diff,
                "WorstChilds1": row1["WorstChilds"],
                "WorstChilds2": row2["WorstChilds"],
                "WorstChildsDiff": worse_diff,
                "PercentImprovementBetter": improvementBetter,
                "PercentImprovementWorse": improvementWorse,
            }
        )

    return pd.DataFrame(results)

def plot_top_parameter_sets(df, top_n=5):
    """
    Plots a bar chart with the top N parameter sets based on the mean fitness

    @param df: DataFrame containing the combined data for analysis
    @param top_n: Number of top parameter sets to plot
    """
    relevant_columns = [
        "ExperimentId",
        "Generations",
        "CrossoverRate",
        "MutationRate",
        "TournamentSize",
        "ElitismEnabled",
        "MeanFitnessOnTraining",
        "BestFitness",
        "WorstFitness"
    ]

    # Group by the relevant columns and calculate the mean fitness
    grouped = df[relevant_columns].groupby(
        [
            "Generations",
            "CrossoverRate",
            "MutationRate",
            "TournamentSize",
            "ElitismEnabled",
        ]
    ).agg(
        MeanFitness=("MeanFitnessOnTraining", "mean"),
        BestFitness=("BestFitness", "max"),
        WorstFitness=("WorstFitness", "min")
    ).reset_index()

    grouped["ExperimentId"] = df["ExperimentId"]

    sorted_group = grouped.sort_values("MeanFitness", ascending=False)

    top_n_group = sorted_group.head(top_n)

    # Plots
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(top_n_group["ExperimentId"].astype(str), top_n_group["MeanFitness"], color='blue', label='Fitness Média')

    ax.plot(top_n_group["ExperimentId"].astype(str), top_n_group["BestFitness"], label='Melhor Fitness', color='green', marker='o', linestyle='--')

    ax.plot(top_n_group["ExperimentId"].astype(str), top_n_group["WorstFitness"], label='Pior Fitness', color='red', marker='x', linestyle='--')

    ax.set_title(f"Top {top_n} Conjuntos de Parâmetros - Fitness Média, Melhor e Pior Fitness", fontsize=14)
    ax.set_xlabel("ID do Conjunto de Parâmetros (ExperimentId)", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)

    ax.legend()

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.xticks(rotation=0, fontsize=9)

    plt.tight_layout()

    plt.savefig(os.path.join(FIGS_FOLDER, f"top_{top_n}_parameter_sets{FIGS_FORMAT}"), dpi=FIGS_DPI)


def get_extreme_fitness(df):
    """
    Get the rows with the highest and lowest fitness values

    @param df: DataFrame containing the combined data for analysis
    """
    max_training_fitness_row = df.loc[df['BestFitnessOnTraining'].idxmax()]
    min_training_fitness_row = df.loc[df['BestFitnessOnTraining'].idxmin()]

    max_test_fitness_row = df.loc[df['BestFitnessOnTest'].idxmax()]
    min_test_fitness_row = df.loc[df['BestFitnessOnTest'].idxmin()]

    return {
        "MaxTrainingFitness": max_training_fitness_row,
        "MinTrainingFitness": min_training_fitness_row,
        "MaxTestFitness": max_test_fitness_row,
        "MinTestFitness": min_test_fitness_row
    }
