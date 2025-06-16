#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/workspaces/tagging_causality
export CUDA_VISIBLE_DEVICES="3,4"
export HDF5_USE_FILE_LOCKING=FALSE

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

identifiers=("bnlearn_child" "bnlearn_earthquake" "bnlearn_insurance" "bnlearn_survey" "bnlearn_asia" "bnlearn_cancer" "bnlearn_alarm" "lucas" "bnlearn_hepar2" "bnlearn_win95pts" "bnlearn_hailfinder")
llms=("Llama-3.3-70B-Instruct" "claude-3-5-sonnet-20241022" "gpt-4-0613" "gpt-4o-2024-08-06" "Qwen2.5-72B-Instruct")

order_data="random"
nr_samples=10000
seeds=(0 1 2 3 4 5 6 7 8 9)
min_samples=(1 2)
remove_singular_tags=("True" "False")
prior_on_weight=("True" "False")
always_meeks=("True" "False")
redirect_existing_edges=("True" "False")
redirecting_strategy=(0 1)
include_current_edge_as_evidence=("True" "False")
include_redirected_edges_in_edge_count=("True")

declare -A identifier_times
start_time=$(date +%s)

for min_sample in "${min_samples[@]}"; do
    for remove_singular in "${remove_singular_tags[@]}"; do
        for llm in "${llms[@]}"; do
            for prior in "${prior_on_weight[@]}"; do
                for always_meek in "${always_meeks[@]}"; do
                    for redirect in "${redirect_existing_edges[@]}"; do
                        if [ "$redirect" == "True" ]; then
                            for strategy in "${redirecting_strategy[@]}"; do
                                for include_current in "${include_current_edge_as_evidence[@]}"; do
                                    for include_redirected in "${include_redirected_edges_in_edge_count[@]}"; do
                                        for identifier in "${identifiers[@]}"; do
                                            for seed in "${seeds[@]}"; do
                                                python run.py \
                                                    --experimental_series "final_eval" \
                                                    --identifier "$identifier" \
                                                    --order_data "random" \
                                                    --seed "$seed" \
                                                    --tagging_approach_id 0 \
                                                    --load_with_llm "$llm" \
                                                    --nr_samples "$nr_samples" \
                                                    --pc_indep_test "chisq" \
                                                    --pc_alpha 0.05 \
                                                    --min_samples "$min_sample" \
                                                    --min_prob_threshold 0.5 \
                                                    --anti_tags "False" \
                                                    --remove_duplicates "True" \
                                                    --remove_singular_tags "$remove_singular" \
                                                    --prior_on_weight "$prior" \
                                                    --always_meek "$always_meek" \
                                                    --redirect_existing_edges "$redirect" \
                                                    --redirecting_strategy "$strategy" \
                                                    --min_prob_redirecting 0.6 \
                                                    --include_current_edge_as_evidence "$include_current" \
                                                    --include_redirected_edges_in_edge_count "$include_redirected" &
                                            done
                                            # if [ $? -ne 0 ]; then
                                            #     echo "Error: The script run.py failed for identifier $identifier, seed $seed, prior $prior, redirect $redirect, strategy $strategy, include_current $include_current, include_redirected $include_redirected, and LLM $llm"
                                            #     exit 1
                                            # fi
                                        done
                                        wait
                                    done
                                done
                            done
                        else
                            for identifier in "${identifiers[@]}"; do
                                # redirecting parameters don't matter here (use default values)
                                for seed in "${seeds[@]}"; do
                                    python run.py \
                                        --experimental_series "final_eval" \
                                        --identifier "$identifier" \
                                        --order_data "random" \
                                        --seed "$seed" \
                                        --tagging_approach_id 0 \
                                        --load_with_llm "$llm" \
                                        --nr_samples "$nr_samples" \
                                        --pc_indep_test "chisq" \
                                        --pc_alpha 0.05 \
                                        --min_samples "$min_sample" \
                                        --min_prob_threshold 0.5 \
                                        --anti_tags "False" \
                                        --remove_duplicates "True" \
                                        --remove_singular_tags "$remove_singular" \
                                        --prior_on_weight "$prior" \
                                        --always_meek "$always_meek" \
                                        --redirect_existing_edges "$redirect" &
                                done
                                # if [ $? -ne 0 ]; then
                                #     echo "Error: The script run.py failed for identifier $identifier, seed $seed, prior $prior, redirect $redirect, strategy $strategy, include_current $include_current, include_redirected $include_redirected, and LLM $llm"
                                #     exit 1
                                # fi
                            done
                            wait
                        fi
                    done
                done
            done
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Total computation time: $total_time seconds"