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

- Existe uma relação entre o RMSE e a correlação de Spearman? Isso pode ajudar a
  identificar a qualidade da generalização

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

import pandas as pd
import os
from typing import Tuple

from matplotlib import pyplot as plt
import seaborn as sns

from .summarizer import OUTPUT_FOLDER, TRAINING_SUMMARY_CSV, RANKING_SUMMARY_CSV

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the summarized data from the summarizer.py script

    @return: A tuple containing the training data and the ranking data
    """
    training_data = pd.read_csv(os.path.join(OUTPUT_FOLDER, TRAINING_SUMMARY_CSV), sep="|")
    ranking_data = pd.read_csv(os.path.join(OUTPUT_FOLDER, RANKING_SUMMARY_CSV), sep="|")

    return training_data, ranking_data

def analyze_discrepancy_and_fitness(ranking_df, training_df):
    """
    Responde à Pergunta #1: Identifica combinações de parâmetros que minimizam a discrepância
    entre rankings e mantêm bom desempenho em fitness absoluto.
    Também plota um heatmap com as métricas relevantes.

    Args:
        ranking_df (pd.DataFrame): DataFrame com os dados de ranking.
        training_df (pd.DataFrame): DataFrame com os dados de treinamento.

    Returns:
        pd.DataFrame: Combinações de parâmetros com melhor generalização.
    """
    # Combinar os dados usando ExperimentId
    combined_data = pd.merge(ranking_df, training_df, on="ExperimentId", suffixes=("_ranking", "_training"))

    # Calcular uma métrica de discrepância
    combined_data["Discrepancy"] = abs(combined_data["BestFitnessOnTest"] - combined_data["BestFitnessOnTraining"])

    # Filtrar combinações que minimizam a discrepância
    min_discrepancy = combined_data["Discrepancy"].min()
    best_discrepancy_data = combined_data[combined_data["Discrepancy"] == min_discrepancy]

    # Verificar bom desempenho em fitness absoluto
    # Critérios: BestFitnessOnTest e BestFitnessOnTraining devem estar entre os top 20%
    test_threshold = combined_data["BestFitnessOnTest"].quantile(0.8)
    training_threshold = combined_data["BestFitnessOnTraining"].quantile(0.8)

    best_generalization = best_discrepancy_data[
        (best_discrepancy_data["BestFitnessOnTest"] >= test_threshold) &
        (best_discrepancy_data["BestFitnessOnTraining"] >= training_threshold)
    ]

    # Criar heatmap com SpearmansCorrelation e Discrepancy
    plt.figure(figsize=(10, 6))
    heatmap_data = combined_data.pivot(
        index="ExperimentId",
        columns="SpearmansCorrelation",
        values="Discrepancy"
    )
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Discrepancy"}
    )
    plt.title("Heatmap: Discrepância vs Spearman's Correlation")
    plt.xlabel("Spearman's Correlation")
    plt.ylabel("ExperimentId")
    plt.tight_layout()
    plt.show()

    return best_generalization[[
        "ExperimentId",
        "Discrepancy",
        "BestFitnessOnTest",
        "BestFitnessOnTraining",
        "RMSE",
        "SpearmansCorrelation"
    ]].sort_values(by="Discrepancy")
