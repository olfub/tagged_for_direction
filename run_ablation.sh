#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/workspaces/tagging_causality
export CUDA_VISIBLE_DEVICES="3,4"
export HDF5_USE_FILE_LOCKING=FALSE

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# undirect, remove, inverse
llms=("gpt-4-0613" "Llama-3.3-70B-Instruct" "claude-3-5-sonnet-20241022" "gpt-4o-2024-08-06" "Qwen2.5-72B-Instruct")
types=("undirect" "remove" "inverse")
param_nrs=(1 2 3 4 5 6)


declare -A identifier_times
start_time=$(date +%s)

for llm in "${llms[@]}"; do
    for type in "${types[@]}"; do
        for param_nr in "${param_nrs[@]}"; do
            python ablation_direction.py --param "$param_nr" --type "$type" --llm "$llm" &
        done
    done
    wait
done

# tags
llms=("gpt-4-0613" "Llama-3.3-70B-Instruct" "claude-3-5-sonnet-20241022" "gpt-4o-2024-08-06" "Qwen2.5-72B-Instruct")
types=("tags")
param_nrs=(1)
seeds=(0 1 2 3 4 5 6 7 8 9)
error_rates=(0.0 0.1 0.2 0.3 0.4 0.5)

for param_nr in "${param_nrs[@]}"; do
    for seed in "${seeds[@]}"; do
        for error_rate in "${error_rates[@]}"; do
            for llm in "${llms[@]}"; do
                for type in "${types[@]}"; do
                    python ablation_direction.py --param "$param_nr" --type "$type" --llm "$llm" --seed "$seed" --error_rate "$error_rate" &
                done
            done
            wait
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Total computation time: $total_time seconds"