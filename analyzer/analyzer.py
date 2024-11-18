#!/usr/bin/env python3

# Filename: analyzer.py
# Created on: November 17, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

"""
This script is responsible for analyzing the data summarized by the summarizer.py script.
"""

"""
Perguntas:

Pergunta #1
- Quais combinações de parâmetros minimizam a discrepância entre ranking
  de dados de teste e treinamento?

- Além de minimizar a discrepância, essas combinações mantêm um bom desempenho em
  fitness absoluto (tanto no treinamento quanto no teste) ?

- Objetivo: encontrar a combinação de parâmetros que resulta em um modelo que
  generaliza bem para dados não vistos

Pergunta #2
- Quais combinações de parâmetros resultam no melhor fitness médio?

- Essas combinações também resultam no melhor fitness médio nos dados de teste?

- Essa combinação resulta em um modelo que generaliza bem, isto é, a discrepância entre
  ranking de dados de teste e treinamento é pequena?

- Objetivo: encontrar a combinação de parâmetros que resulta em um modelo que
  tem o melhor desempenho médio

Pergunta #3
- Há uma correlação entre o melhor fitness no treinamento e o melhor fitness no teste?

- Objetivo: entender se o fitness no treinamento é um bom indicador de fitness no teste

Pergunta #4
- Qual combinação de parâmetros minimizam o RMSE entre os dados de treinamento e teste?

- Objetivo: encontrar a combinação de parâmetros que resulta em um modelo que
  generaliza bem para dados não vistos

Pergunta #5
- Qual combinação de parâmetros geram os melhores genes em termos de fitness?

- Objetivo: entender quais parâmetros resultam em genes que têm um bom desempenho

Pergunta #6
- Aumentar a taxa de mutação melhora o fitness médio?

- Objetivo: entender se a mutação é um fator importante para melhorar o fitness

Pergunta #7
- Aumentar a taxa de mutação melhora a geração de filhos com melhor fitness ou piora?

- Objetivo: entender se a mutação é um fator importante para melhorar o fitness

Pergunta #8
- Aumentar a taxa de cruzamento melhora o fitness médio?

- Objetivo: entender se o cruzamento é um fator importante para melhorar o fitness

Pergunta #9
- O desvio padrão do fitness (StdFitness) diminui à medida que o número de gerações
  aumenta? Isso indica que o algoritmo está convergindo?

- Se o desvio padrão do fitness diminui, o melhor fitness na população também converge?
  Isso está relacionado a overfitting?

- Objetivo: entender se o algoritmo está convergindo

Pergunta #10
- Habilitar elitismo melhora o fitness médio?

- Objetivo: entender se o elitismo é um fator importante para melhorar o fitness

Pergunta #11
- Aumentar o tamanho da população melhora o fitness médio?

- Objetivo: entender se o tamanho da população é um fator importante para melhorar o fitness

Pergunta #12
- Aumentar o número de gerações melhora o fitness médio?

- Objetivo: entender se o número de gerações é um fator importante para melhorar o fitness

Pergunta #13
- Aumentar a pressão seletiva melhora o fitness médio?

- Objetivo: entender se a pressão seletiva é um fator importante para melhorar o fitness

Pergunta #14
- Aumentar a pressão seletiva melhora a geração de filhos com melhor fitness ou piora?

- Objetivo: entender se a pressão seletiva é um fator importante para melhorar o fitness
"""

import os
import pandas as pd
import seaborn as sns

from typing import Tuple
from matplotlib import pyplot as plt

from .constants import (
    OUTPUT_FOLDER,
    TRAINING_SUMMARY_CSV,
    RANKING_SUMMARY_CSV,
    FIGS_FOLDER,
    FIGS_FORMAT,
    FIGS_DPI,
    FIGS_SIZE,
    print_line,
    print_separator,
)


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

def plot_fitness_vs_paramter(df, parameter_name):
    """
    Responde à Pergunta #5: Qual combinação de parâmetros geram os melhores genes em termos de fitness?

    @param df: DataFrame com os dados
    @param parameter_name: Nome do parâmetro a ser plotado
    """
    pivot_table = df.pivot_table(
        values="BestFitnessOnTraining", 
        index=parameter_name, 
        aggfunc="mean"
    )
    
    plt.figure(figsize=FIGS_SIZE)
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)

    plt.xlabel(parameter_name)
    plt.ylabel("Best Fitness on Training")
    plt.title(f"Heatmap of Best Fitness on Training vs {parameter_name}")

    plt.savefig(
        os.path.join(FIGS_FOLDER, f"heatmap_fitness_vs_{parameter_name}.{FIGS_FORMAT}"),
        dpi=FIGS_DPI,
    )
    plt.close()


def plot_correlation_between_training_and_test(df):
    """
    Responde à Pergunta #3: Há uma correlação entre o melhor fitness no treinamento e o melhor fitness no teste?

    @param df: DataFrame com os dados
    """
    print("Plotting correlation between training and test...")

    plt.figure(figsize=FIGS_SIZE)

    sns.scatterplot(data=df, x="BestFitnessOnTraining", y="BestFitnessOnTest")

    plt.xlabel("Best Fitness on Training")
    plt.ylabel("Best Fitness on Test")
    plt.title("Correlation between Best Fitness on Training and Test")

    plt.savefig(
        os.path.join(
            FIGS_FOLDER, f"correlation_between_training_and_test.{FIGS_FORMAT}"
        ),
        dpi=FIGS_DPI,
    )

    plt.close()

    gene_with_best_fitness_on_training = df.loc[df["BestFitnessOnTraining"].idxmax()]

    gene_with_best_fitness_on_test = df.loc[df["BestFitnessOnTest"].idxmax()]

    print("Best fitness on training:")
    print(gene_with_best_fitness_on_training)
    print_separator()
    print("Best fitness on test:")
    print(gene_with_best_fitness_on_test)


def plot_fitness_vs_parameter(df, parameter_name):
    """
    Responde à Pergunta #5: Qual combinação de parâmetros geram os melhores genes em termos de fitness?

    @param df: DataFrame com os dados
    @param parameter_name: Nome do parâmetro a ser plotado
    """
    plt.figure(figsize=FIGS_SIZE)

    sns.scatterplot(data=df, x=parameter_name, y="BestFitnessOnTraining")
    plt.xlabel(parameter_name)
    plt.ylabel("Best Fitness on Training")
    plt.title(f"Fitness vs. {parameter_name}")

    plt.savefig(
        os.path.join(FIGS_FOLDER, f"fitness_vs_{parameter_name}.{FIGS_FORMAT}"),
        dpi=FIGS_DPI,
    )


def plot_fitness_vs_mutation_rate(df):
    """
    Responde à Pergunta #6: Aumentar a taxa de mutação melhora o fitness médio?

    @param df: DataFrame com os dados
    """
    plot_fitness_vs_parameter(df, "MutationRate")


def plot_fitness_vs_crossover_rate(df):
    """
    Responde à Pergunta #8: Aumentar a taxa de cruzamento melhora o fitness médio?

    @param df: DataFrame com os dados
    """
    plot_fitness_vs_parameter(df, "CrossoverRate")


def plot_fitness_vs_population_size(df):
    """
    Responde à Pergunta #11: Aumentar o tamanho da população melhora o fitness médio?

    @param df: DataFrame com os dados
    """
    plot_fitness_vs_parameter(df, "PopulationSize")


def plot_fitness_vs_generations(df):
    """
    Responde à Pergunta #12: Aumentar o número de gerações melhora o fitness médio?

    @param df: DataFrame com os dados
    """
    plot_fitness_vs_parameter(df, "Generations")


def plot_fitness_vs_tournament_size(df):
    """
    Responde à Pergunta #13: Aumentar a pressão seletiva melhora o fitness médio?

    @param df: DataFrame com os dados
    """
    plot_fitness_vs_parameter(df, "TournamentSize")


def plot_fitness_vs_elitism_enabled(df):
    """
    Responde à Pergunta #10: Habilitar elitismo melhora o fitness médio?

    @param df: DataFrame com os dados
    """
    plot_fitness_vs_parameter(df, "ElitismEnabled")


def plot_fitness_vs_std_fitness(df):
    """
    Responde à Pergunta #9: O desvio padrão do fitness (StdFitness) diminui à medida que o número de gerações aumenta?

    @param df: DataFrame com os dados de treinamento
    """
    plt.figure(figsize=FIGS_SIZE)

    sns.scatterplot(data=df, x="Generations", y="StdFitness")
    plt.xlabel("Generations")
    plt.ylabel("StdFitness")
    plt.title("StdFitness vs. Generations")

    plt.savefig(
        os.path.join(FIGS_FOLDER, f"std_fitness_vs_generations.{FIGS_FORMAT}"),
        dpi=FIGS_DPI,
    )


def plot_fitness_vs_mean_fitness(df):
    """
    Responde à Pergunta #9: O desvio padrão do fitness (StdFitness) diminui à medida que o número de gerações aumenta?

    @param df: DataFrame com os dados
    """
    plt.figure(figsize=FIGS_SIZE)

    sns.scatterplot(data=df, x="Generations", y="MeanFitness")
    plt.xlabel("Generations")
    plt.ylabel("MeanFitness")
    plt.title("MeanFitness vs. Generations")

    plt.savefig(
        os.path.join(FIGS_FOLDER, f"mean_fitness_vs_generations.{FIGS_FORMAT}"),
        dpi=FIGS_DPI,
    )


def plot_fitness_vs_median_fitness(df):
    """
    Responde à Pergunta #9: O desvio padrão do fitness (StdFitness) diminui à medida que o número de gerações aumenta?

    @param df: DataFrame com os dados
    """
    plt.figure(figsize=FIGS_SIZE)

    sns.scatterplot(data=df, x="Generations", y="MedianFitness")
    plt.xlabel("Generations")
    plt.ylabel("MedianFitness")
    plt.title("MedianFitness vs. Generations")

    plt.savefig(
        os.path.join(FIGS_FOLDER, f"median_fitness_vs_generations.{FIGS_FORMAT}"),
        dpi=FIGS_DPI,
    )


def get_parameter_combinations_with_less_ranking_discrepancy(df):
    """
    Responde à Pergunta #1: Quais combinações de parâmetros minimizam a discrepância entre ranking de dados de teste e treinamento?

    @param df: DataFrame com os dados
    @return: DataFrame com as melhores combinações de parâmetros
    """
    return df.loc[df["SpearmansCorrelation"].idxmax()]


def get_parameter_combinations_with_more_ranking_discrepancy(df):
    """
    Responde à Pergunta #1: Quais combinações de parâmetros minimizam a discrepância entre ranking de dados de teste e treinamento?

    @param df: DataFrame com os dados
    @return: DataFrame com as piores combinações de parâmetros
    """
    return df.loc[df["SpearmansCorrelation"].idxmin()]


def get_parameter_combinations_with_best_fitness_on_test(df):
    """
    Responde à Pergunta #2: Identifica as combinações de parâmetros que resultam no melhor fitness médio

    @param df: DataFrame com os dados
    @return: DataFrame com as melhores combinações de parâmetros
    """
    # Identificar a combinação de parâmetros com melhor fitness médio
    best_fitness_combination = df.loc[df["MeanFitnessOnTest"].idxmax()]

    return best_fitness_combination

def get_parameter_combinations_with_worst_fitness_on_test(df):
    """
    Responde à Pergunta #2: Identifica as combinações de parâmetros que resultam no pior fitness médio

    @param df: DataFrame com os dados
    @return: DataFrame com as piores combinações de parâmetros
    """
    # Identificar a combinação de parâmetros com pior fitness médio
    worst_fitness_combination = df.loc[df["MeanFitnessOnTest"].idxmin()]

    return worst_fitness_combination
