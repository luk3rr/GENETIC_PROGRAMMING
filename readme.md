# Genetic Algorithm Simulation

Este projeto implementa uma simulação de algoritmo genético para encontra a melhor função de distância de um conjunto de testes. Ele possui dois módulos principais: `gp` e `analyzer`.

## Módulo `gp`

O módulo `gp` executa a simulação do algoritmo genético. Ele aceita uma série de parâmetros que controlam a execução da simulação, como o tamanho da população, número de gerações, probabilidades de crossover e mutação, entre outros.

### Como usar

Para verificar os argumentos disponíveis, execute o seguinte comando:

```bash
python3 -m gp -h
```

### Argumentos

- `-h`, `--help`: Exibe esta mensagem de ajuda e sai.
- `-s SEED`, `--seed SEED`: Semente para a geração de números aleatórios.
- `-ts THREAD_INITIAL_SEED`, `--thread-initial-seed THREAD_INITIAL_SEED`: Semente inicial para as threads.
- `-p POPULATION_SIZE`, `--population-size POPULATION_SIZE`: Tamanho da população.
- `-c CROSSOVERS_BY_GENERATION`, `--crossovers-by-generation CROSSOVERS_BY_GENERATION`: Número de crossovers por geração. O valor padrão é metade do tamanho da população.
- `-g NUM_GENERATIONS`, `--num-generations NUM_GENERATIONS`: Número de gerações.
- `-cp CROSSOVER_PROB`, `--crossover-prob CROSSOVER_PROB`: Probabilidade de crossover.
- `-mp MUTATION_PROB`, `--mutation-prob MUTATION_PROB`: Probabilidade de mutação.
- `-t TOURNAMENT_SIZE`, `--tournament-size TOURNAMENT_SIZE`: Tamanho do torneio.
- `-e ELITISM_SIZE`, `--elitism-size ELITISM_SIZE`: Tamanho do elitismo como uma fração do tamanho da população. O valor padrão é 5% do tamanho da população.
- `-i SIMULATION_ID`, `--simulation-id SIMULATION_ID`: Identificador da simulação. Usado para salvar os dados da simulação. O valor padrão é o timestamp atual.
- `-d DATASET`, `--dataset DATASET`: Dataset a ser usado. As opções são 'BCC' e 'WR'.
- `-w WORKERS`, `--workers WORKERS`: Número de threads para processar em paralelo. Se não especificado, apenas o processo principal será utilizado.

Exemplo de execução:

```bash
python3 -m gp -p 100 -g 50 -s 42 -d BCC -w 4
```

Este comando executará a simulação com uma população de 100 indivíduos, 50 gerações, uma semente de 42, usando o dataset 'BCC' e 4 núcleos para processamento paralelo.

---

## Módulo `analyzer`

O módulo `analyzer` é usado para analisar e agregar os dados das simulações realizadas com o módulo `gp`. Ele não recebe argumentos e é executado da seguinte maneira:

### Como usar

Para rodar o analisador, execute o seguinte comando:

```bash
python3 -m analyzer
```

Este módulo irá buscar e processar os dados das simulações salvas, fornecendo análises e resultados agregados.

---

## Requisitos

Para rodar este projeto, você precisará do Python 3.x e das dependências descritas no arquivo `requirements.txt`. Para instalá-las, basta executa:

```bash
pip install -r requirements.txt
```
