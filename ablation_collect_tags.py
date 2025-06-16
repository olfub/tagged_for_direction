import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plots(ablation_type):
    llm_data = {}
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for llm in llms:
        data = {}
        for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            data_seed = {}
            for seed in seeds:
                # load csv
                df = pd.read_csv(f"{type_path}1_{seed}_{i}/{llm}.csv")
                for dataset in datasets:
                    correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                    incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                    undirected = df.loc[df['dataset'] == dataset, 'tagging_nothing'].sum()
                    if dataset not in data_seed:
                        data_seed[dataset] = []
                    data_seed[dataset].append((correct, incorrect, undirected))
            for dataset in datasets:
                if dataset not in data:
                    data[dataset] = []
                average_correct = np.mean([x[0] for x in data_seed[dataset]])
                average_incorrect = np.mean([x[1] for x in data_seed[dataset]])
                average_undirected = np.mean([x[2] for x in data_seed[dataset]])
                data[dataset].append(average_correct/(average_correct+average_incorrect))

        # make plots

        # filter out datasets with only NaN values
        filtered_data = {dataset: accuracies for dataset, accuracies in data.items() if not all(np.isnan(accuracies))}
        filtered_legend = [dataset_names[dataset] for dataset in filtered_data.keys()]

        plt.figure(figsize=(14, 7))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X']
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:11]
        for idx, dataset in enumerate(datasets):
            if dataset not in filtered_data:
                continue	
            accuracies = data[dataset]
            color = default_colors[idx % len(default_colors)]
            marker = markers[idx % len(markers)]
            markersize = 20 if marker == '*' else 15
            plt.plot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], accuracies, marker=marker, label=dataset_names[dataset], markersize=markersize, linewidth=5, zorder=10, color=color)
            plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ['0%', '10%', '20%', '30%', '40%', '50%'])

        plt.xlabel('Tag Error Percentage', fontsize=30)
        plt.ylabel('Accuracy', fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.grid(True)
        
        plt.legend(filtered_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=26)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{llm}.pdf")
        plt.clf()

        llm_data[llm] = data

    # Calculate average accuracy over all LLMs
    average_data = {dataset: [] for dataset in datasets}

    for dataset in datasets:
        for i in range(6):
            accuracies = [llm_data[llm][dataset][i] for llm in llms if dataset in llm_data[llm]]
            average_accuracy = np.nanmean(accuracies)
            average_data[dataset].append(average_accuracy)

    # make plots
    plt.figure(figsize=(14, 7))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X']
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:11]
    # Plot average accuracy
    filtered_legend = []
    for idx, dataset in enumerate(datasets):
        if all(np.isnan(average_data[dataset])):
            continue
        filtered_legend.append(dataset_names[dataset])
        accuracies = average_data[dataset]
        color = default_colors[idx % len(default_colors)]
        marker = markers[idx % len(markers)]
        markersize = 20 if marker == '*' else 15
        plt.plot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], accuracies, marker=marker, label=dataset_names[dataset], markersize=markersize, linewidth=5, zorder=10, color=color)

    plt.plot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [np.nanmean([average_data[dataset][i] for dataset in datasets]) for i in range(6)], marker='X', label='Average', markersize=25, linewidth=5, zorder=10, color="black")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ['0%', '10%', '20%', '30%', '40%', '50%'])

    plt.xlabel('Tag Error Percentage', fontsize=30)
    plt.ylabel('Average Accuracy', fontsize=30)
    # plt.title('Average Accuracy vs Tag Error Percentage', fontsize=30)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)

    plt.legend(filtered_legend + ["All"], loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=26)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/average.pdf")
    print(f"{save_path}/average.pdf")
    plt.clf()

base_path = "results/ablation/"
base_save_path = "plots/ablation/"
datasets = [
    "bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", 
    "bnlearn_asia", "lucas", "bnlearn_child", 
    "bnlearn_insurance", "bnlearn_alarm", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"
]
dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                    "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                    "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}

llms = ["Llama-3.3-70B-Instruct", "claude-3-5-sonnet-20241022", "gpt-4-0613", "gpt-4o-2024-08-06", "Qwen2.5-72B-Instruct"]
llm_to_text = {"Llama-3.3-70B-Instruct": "Llama-3.3", "claude-3-5-sonnet-20241022": "Claude-3.5", "gpt-4-0613": "GPT-4", "gpt-4o-2024-08-06": "GPT-4o", "Qwen2.5-72B-Instruct": "Qwen-2.5"}

plots("tags")