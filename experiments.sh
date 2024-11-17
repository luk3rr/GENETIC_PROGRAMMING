#!/usr/bin/env bash

# Filename: experiments.sh
# Created on: November 15, 2024
# Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

PROGRAM_OUTPUT="output.txt"

PROGRAM="python3 ./main.py"

DATASETS="BCC" # WR
SEEDS="381221029 123456789 987654321 135792468 246813579 102938475 564738291 819273645 111213141 151617181"
THREAD_INITIAL_SEED="40028922"
ENABLE_ELITISM="1 0"
POPULATION_SIZES="100 200" #"25 50 100 200"
GENERATIONS="10 20 40 80"
CROSSOVER_RATES="0.9 0.6"
MUTATION_RATES="0.05 0.3"
TOURNAMENT_SIZES="3 7"

LOG_FOLDER="log/"

print_line() {
    echo "--------------------------------------------------------------------"
}

total_experiments=$((
    $(echo "$DATASETS" | wc -w) * \
    $(echo "$SEEDS" | wc -w) * \
    $(echo "$ENABLE_ELITISM" | wc -w) * \
    $(echo "$POPULATION_SIZES" | wc -w) * \
    $(echo "$GENERATIONS" | wc -w) * \
    $(echo "$CROSSOVER_RATES" | wc -w) * \
    $(echo "$MUTATION_RATES" | wc -w) * \
    $(echo "$TOURNAMENT_SIZES" | wc -w)
))

# Contador de experimentos
current_experiment=0

run_experiment() {
	seed=$1
	population=$2
	generations=$3
	pc=$4
	pm=$5
	tournament_size=$6
	enable_elitism=$7
	data=$8

	salt=$(date +%s)

	identifier_no_salt="SD${seed}_PS${population}_GS${generations}_PC${pc}_PM${pm}_TS${tournament_size}_EE${enable_elitism}"

    output_pattern="${LOG_FOLDER}${identifier_no_salt}*.dat"

    # Check if the experiment already ran
    if compgen -G "$output_pattern" > /dev/null; then
        echo "Experiment $identifier_no_salt already ran. Skipping..."
        return
    fi

	identifier="${identifier_no_salt}_SALT${salt}"

	echo "Experiment id: $identifier"

	if [ "$enable_elitism" -eq 1 ]; then
		elitism_option=()
	else
		elitism_option=(-e 0)
	fi

    start_time=$(date +%s)

	$PROGRAM \
		-s "$seed" \
		-ts "$THREAD_INITIAL_SEED" \
		-p "$population" \
		-g "$generations" \
		-cp "$pc" \
		-mp "$pm" \
		-t "$tournament_size" \
        -d "$data" \
        "${elitism_option[@]}" -w 12 \
		-i "$identifier" >> "$PROGRAM_OUTPUT" 2> >(tee -a /dev/stderr)

    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))

    echo "Experiment $identifier ran in $elapsed_time seconds."
}

start_time=$(date +%s)

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        for elitism in $ENABLE_ELITISM; do
            for population in $POPULATION_SIZES; do
                for generations in $GENERATIONS; do
                    for crossover_rates in $CROSSOVER_RATES; do
                        for mutation_rates in $MUTATION_RATES; do
                            for tournament_size in $TOURNAMENT_SIZES; do
                                current_experiment=$((current_experiment + 1))

                                echo "Running experiment $current_experiment of $total_experiments"

                                run_experiment "$seed" "$population" "$generations" "$crossover_rates" \
                                    "$mutation_rates" "$tournament_size" "$elitism" "$dataset"

                                print_line
                            done
                        done
                    done
                done
            done
        done
    done
done

end_time=$(date +%s)

echo "All experiments finished!"
echo "Total time elapsed: $((end_time - start_time)) seconds."
