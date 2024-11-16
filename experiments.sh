#!/usr/bin/env bash

# Filename: experiments.sh
# Created on: November 15, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

PROGRAM_OUTPUT="output.txt"

PROGRAM="python3 ./main.py"

SEEDS="381221029 123456789 987654321 135792468 246813579 102938475 564738291 819273645 111213141 151617181"
THREAD_INITIAL_SEED="40028922 "
ENABLE_ELITISM="1 0"
POPULATION_SIZES="50 100 200 400"
GENERATIONS="10 50 100 200"
CROSSOVER_RATES="0.9 0.6"
MUTATION_RATES="0.05 0.3"
TOURNAMENT_SIZES="2 3 5 7"

run_experiment() {
	seed=$1
	population=$2
	generations=$3
	pc=$4
	pm=$5
	tournament_size=$6
	enable_elitism=$7

	identifier="SD${seed}_PS${population}_GS${generations}_PC${pc}_PM${pm}_TS${tournament_size}_EE${enable_elitism}"

	echo "Running experiment: $identifier"
	echo "---------------------------------------------------------\
		----------------------------------------" >> $PROGRAM_OUTPUT
	echo "Running experiment: $identifier" >> $PROGRAM_OUTPUT

	if [ "$enable_elitism" -eq 1 ]; then
		elitism_option="-i $identifier"
	else
		elitism_option="-i $identifier -e 0"
	fi

	$PROGRAM \
		-s "$seed" \
		-ts "$THREAD_INITIAL_SEED" \
		-p "$population" \
		-g "$generations" \
		-cp "$pc" \
		-mp "$pm" \
		-t "$tournament_size" \
		"$elitism_option" >> "$PROGRAM_OUTPUT" 2> >(tee -a /dev/stderr)
}

for seed in $SEEDS; do
	for elitism in $ENABLE_ELITISM; do
		for population in $POPULATION_SIZES; do
			for generations in $GENERATIONS; do
				for crossover_rates in $CROSSOVER_RATES; do
					for mutation_rates in $MUTATION_RATES; do
						for tournament_size in $TOURNAMENT_SIZES; do
							run_experiment "$seed" "$population" "$generations" "$crossover_rates" \
								"$mutation_rates" "$tournament_size" "$elitism"
						done
					done
				done
			done
		done
	done
done
echo "All experiments finished!"
