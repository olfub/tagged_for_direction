import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def table(ablation_type, seed):
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for llm in llms:
        data = {}
        for i in range(1, 7):
            # load csv
            df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
            for dataset in datasets:
                correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                undirected = df.loc[df['dataset'] == dataset, 'tagging_nothing'].sum()
                # accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                # samples = correct + incorrect
                if dataset not in data:
                    data[dataset] = []
                # data[dataset].append((accuracy, samples))
                data[dataset].append((correct, incorrect, undirected))

        def acc_string(value, sample):
            if np.isnan(value):
                return "-"
            value_str = f"{value:.2f}"
            if sample == 1:
                value_str += "*"
            elif sample < 5:
                value_str += "$^\circ$"
            return value_str
        
        def values_string(correct, incorrect, undirected):
            correct_str = str(correct)
            incorrect_str = str(incorrect)
            undirected_str = str(undirected)
            return f"{correct_str} / {incorrect_str} / {undirected_str}"

        # make table
        with open(f"{save_path}/{llm}.txt", "w") as f:
            f.write("\\begin{tabular}{l|ccccccccccc}\n")
            data_str = " & ".join([dataset_names[dataset][:2] for dataset in datasets])
            f.write(f" & {data_str} \\\\\n")
            f.write("\\hline\n")
            for i in range(1, 7):
                # accuracies = [data[dataset][i - 1] for dataset in datasets]
                # accuracies_str = " & ".join([acc_string(accuracy, sample) for accuracy, sample in accuracies])
                # f.write(f"{i} & {accuracies_str} \\\\\n")
                values = [data[dataset][i - 1] for dataset in datasets]
                values_str = " & ".join([values_string(correct, incorrect, undirected) for correct, incorrect, undirected in values])
                f.write(f"{i} & {values_str} \\\\\n")
            f.write("\\end{tabular}\n")

    # and one big table that include all llms
    with open(f"{save_path}/all.txt", "w") as f:
        f.write("\\begin{tabular}{cl|ccccccccccc}\n")
        data_str = " & ".join([dataset_names[dataset][:2] for dataset in datasets])
        f.write(f" & & {data_str} \\\\\n")
        f.write("\\hline\n")
        for llm in llms:
            data = {}
            for i in range(1, 7):
                # load csv
                df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
                for dataset in datasets:
                    correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                    incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                    undirected = df.loc[df['dataset'] == dataset, 'tagging_nothing'].sum()
                    if dataset not in data:
                        data[dataset] = []
                    data[dataset].append((correct, incorrect, undirected))

            f.write("\\hline\n")
            llm_text = llm_to_text[llm]
            f.write(f"\\multirow{{6}}{{*}}{{\\rotatebox{{90}}{{{llm_text}}}}}")
            for i in range(1, 7):
                values = [data[dataset][i - 1] for dataset in datasets]
                values_str = " & ".join([values_string(correct, incorrect, undirected) for correct, incorrect, undirected in values])
                f.write(f" & {i} & {values_str} \\\\\n")
        f.write("\\end{tabular}\n")



def table2(ablation_type, seed):
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f"{save_path}/all_llms.txt", "w") as f:
        f.write("\\begin{tabular}{l|cccccc}\n")
        f.write("LLM & 1 & 2 & 3 & 4 & 5 & 6 \\\\\n")
        f.write("\\hline\n")
        for llm in llms:
            data = {}
            accuracies = []
            for i in range(1, 7):
                # load csv
                df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
                correct = 0
                incorrect = 0
                for dataset in datasets:
                    correct += df.loc[df['dataset'] == dataset, 'tagging_correct'].item()
                    incorrect += df.loc[df['dataset'] == dataset, 'tagging_incorrect'].item()
                accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                accuracies.append(accuracy)

            accuracies_str = " & ".join([f"{accuracy:.2f}" if not np.isnan(accuracy) else "-" for accuracy in accuracies])
            f.write(f"{llm} & {accuracies_str} \\\\\n")
        f.write("\\end{tabular}\n")

def plots(ablation_type, seed):
    llm_data = {}
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for llm in llms:
        data = {}
        for i in range(1, 7):
            # load csv
            df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
            for dataset in datasets:
                correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                if dataset not in data:
                    data[dataset] = []
                data[dataset].append(accuracy)

        # make plots
        for dataset, accuracies in data.items():
            plt.plot(range(1, 7), accuracies, marker='o', label=dataset_names[dataset])

        plt.xlabel('Number Edges')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number Edges')
        plt.legend()
        plt.grid(True)
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

    # Plot average accuracy
    for dataset, accuracies in average_data.items():
        plt.plot(range(1, 7), accuracies, marker='o', label=dataset_names[dataset])

    plt.plot(range(1, 7), [np.nanmean([average_data[dataset][i] for dataset in datasets]) for i in range(6)], marker='o', label='Average')

    plt.xlabel('Undirect Number')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy vs Undirect Number')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/average.pdf")
    plt.clf()


def combined_remove_inverse(llm, seed):
    remove_data = {}
    inverse_data = {}
    
    for ablation_type, data_dict in zip(["remove", "inverse"], [remove_data, inverse_data]):
        type_path = base_path + f"{ablation_type}/"
        data = {}
        for i in range(1, 7):
            # load csv
            df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
            for dataset in datasets:
                correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                if dataset not in data:
                    data[dataset] = []
                data[dataset].append(accuracy)
        data_dict[llm] = data

    # Plot side by side with shared y-axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X']
    for idx, dataset in enumerate(datasets):
        marker = markers[idx % len(markers)]
        markersize = 17 if marker == '*' else 13
        if dataset == "bnlearn_earthquake":
            axes[0].plot(range(1, 7), remove_data[llm][dataset], marker=marker, label=dataset_names[dataset], markersize=markersize, linewidth=3, zorder=10)
            axes[1].plot(range(1, 7), inverse_data[llm][dataset], marker=marker, label=dataset_names[dataset], markersize=markersize, linewidth=3, zorder=10)
        else:
            axes[0].plot(range(1, 7), remove_data[llm][dataset], marker=marker, label=dataset_names[dataset], markersize=markersize, linewidth=3)
            axes[1].plot(range(1, 7), inverse_data[llm][dataset], marker=marker, label=dataset_names[dataset], markersize=markersize, linewidth=3)

    # axes[0].set_title('Fewer Edges', fontsize=34)
    axes[0].set_xlabel('Removed Edges', fontsize=30)
    axes[0].set_ylabel('Accuracy', fontsize=30)
    axes[0].tick_params(axis='both', which='major', labelsize=24)
    axes[0].set_xticks(range(1, 7))
    axes[0].grid(True)

    # axes[1].set_title('Incorrect Edges', fontsize=34)
    axes[1].set_xlabel('Inverted Edges', fontsize=30)
    axes[1].tick_params(axis='both', which='major', labelsize=24)
    axes[1].set_xticks(range(1, 7))
    axes[1].grid(True)

    # Create a single legend for both plots with markers
    handles, labels = axes[0].get_legend_handles_labels()
    scatter_handles = []
    for handle in handles:
        marker = handle.get_marker()
        markersize = 35 if marker == '*' else 25
        scatter_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=handle.get_color(), markersize=markersize))
    
    # Reduce the white space between marker and text
    # fig.legend(scatter_handles[:6], labels[:6], loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=22, handletextpad=0.2)
    # fig.legend(scatter_handles[6:], labels[6:], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.2), fontsize=22, handletextpad=0.2)

    fig.legend(scatter_handles[:4], labels[:4], loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.11), fontsize=26, handletextpad=0.2)
    fig.legend(scatter_handles[4:8], labels[4:8], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.21), fontsize=26, handletextpad=0.2)
    fig.legend(scatter_handles[8:], labels[8:], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.31), fontsize=26, handletextpad=0.2)

    plt.tight_layout()
    plt.savefig(f"{base_save_path}/{llm}_combined_remove_inverse.pdf", bbox_inches='tight')
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
seed=0

for ablation_type in ["undirect", "remove", "inverse"]:
    table(ablation_type, seed=seed)
    table2(ablation_type, seed=seed)
    plots(ablation_type, seed=seed)

for llm in llms:
    combined_remove_inverse(llm, seed=seed)   
