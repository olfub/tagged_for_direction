import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from discovery.pc_tagged import get_edge_count
from util import load_data, load_from_llm, make_deterministic

datasets = [
    "bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", 
    "bnlearn_asia", "lucas", "bnlearn_child", 
    "bnlearn_insurance", "bnlearn_alarm", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"
]
llms = ["Llama-3.3-70B-Instruct", "claude-3-5-sonnet-20241022", "gpt-4-0613", "gpt-4o-2024-08-06", "Qwen2.5-72B-Instruct"]
# llms = ["gpt-4-0613"]

dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                    "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                    "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}

anti_tags = False
remove_duplicates = True
remove_singular_tags = False

min_evidence = 1

accuracies = {}

make_deterministic(0)

# create heatmap for each dataset
for llm in llms:

    for dataset in datasets:
        # load data (graph)
        variables, var_labels, _, edges, positions, _ = load_data(dataset, order_data="random", seed=0)

        # load tags
        data_id = dataset.split("_")[-1] if dataset.startswith("bnlearn") else dataset
        tags = load_from_llm(f"{llm}_tag_{data_id}_True.txt", variables=var_labels, anti_tags=anti_tags, remove_duplicates=remove_duplicates, remove_singular_tags=remove_singular_tags)
        tag_list = [tags[var] for var in var_labels]

        # compute edge count
        unique_tags = list(set([tag for var_tags in tag_list for tag in var_tags]))
        unique_tags.sort()
        edge_count = get_edge_count(unique_tags, tags, edges)

        # edge_count to matrix
        edge_count_matrix = np.zeros((len(unique_tags), len(unique_tags)), dtype=int)
        for edge, count in edge_count.items():
            edge_0 = unique_tags.index(edge[0])
            edge_1 = unique_tags.index(edge[1])
            edge_count_matrix[edge_0][edge_1] = count

        # calculate accuracy
        edge_acc_matrix = np.zeros_like(edge_count_matrix, dtype=float)
        evidence_ones = np.zeros_like(edge_count_matrix, dtype=int)
        evidence_twos = np.zeros_like(edge_count_matrix, dtype=int)
        for i in range(len(unique_tags)):
            for j in range(len(unique_tags)):
                if i < j:
                    continue
                evidence_sum = edge_count_matrix[i][j] + edge_count_matrix[j][i]
                if evidence_sum == 1:
                    evidence_ones[i][j] = 1
                    evidence_ones[j][i] = 1
                elif evidence_sum == 2:
                    evidence_twos[i][j] = 1
                    evidence_twos[j][i] = 1
                if evidence_sum < min_evidence:
                    edge_acc_matrix[i][j] = -1
                    edge_acc_matrix[j][i] = -1
                    continue
                acc = edge_count_matrix[i][j] / (edge_count_matrix[i][j] + edge_count_matrix[j][i])
                if acc < 0.5:
                    acc = 1 - acc
                edge_acc_matrix[i][j] = acc
                edge_acc_matrix[j][i] = acc

        # normalize to [0, 1]
        edge_acc_matrix = (edge_acc_matrix - 0.5) * 2
        edge_acc_matrix[edge_acc_matrix < 0] = np.nan

        # draw two heatmaps
        plt.figure(figsize=(10, 8))

        colors = ["lightgrey", "dodgerblue"]
        cmap = LinearSegmentedColormap.from_list("lightgrey_dodgerblue", colors, N=256)

        # Set grey color for NaN values
        edge_acc_matrix_masked = np.ma.masked_invalid(edge_acc_matrix)
        cmap.set_bad(color='white')
        # ax = sns.heatmap(edge_acc_matrix_masked, xticklabels=unique_tags, yticklabels=unique_tags, cmap=cmap, annot=False, vmin=0, vmax=1)
        ax = sns.heatmap(edge_acc_matrix_masked, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, vmin=0, vmax=1)

        # Add a box around all heatmap elements that are not NaN
        for i in range(len(unique_tags)):
            for j in range(len(unique_tags)):
                if not np.isnan(edge_acc_matrix[i, j]):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=1, zorder=10))

        # Rotate x labels by 45 degrees
        plt.xticks(rotation=45, ha='right')

        # Add hatches
        if len(unique_tags) < 10:
            mpl.rcParams['hatch.linewidth'] = 6
            one_hatch = 'X'
            two_hatch = '/'
        elif len(unique_tags) < 20:
            mpl.rcParams['hatch.linewidth'] = 3
            one_hatch = 'XX'
            two_hatch = '//'
        else:
            mpl.rcParams['hatch.linewidth'] = 1
            one_hatch = 'XXXX'
            two_hatch = '////'
        for i in range(len(unique_tags)):
            for j in range(len(unique_tags)):
                if evidence_ones[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch=one_hatch, edgecolor='white', lw=0))
                elif evidence_twos[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch=two_hatch, edgecolor='white', lw=0))
                # elif np.isnan(edge_acc_matrix[i, j]):
                #     ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='/', edgecolor='black', lw=0))

        # Increase the ticks and the labels on the colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=34, length=20, width=4)

        plt.title(dataset_names[dataset], fontsize=16)
        plt.tight_layout()  # Adjust layout to ensure no label text is cut off
        plt.savefig(f"plots/homogeneity/{llm}_{dataset}_tag_homogenity.pdf")
        plt.close()

        # save accuracy
        avg_accuracy = np.nanmean(edge_acc_matrix)
        accuracies[(llm, dataset)] = avg_accuracy

        if llm == "gpt-4-0613": # and dataset == "bnlearn_alarm":
            with open('plots/homogeneity/gpt4_pairs.txt', 'a') as f:
                f.write(f" & \\textbf{{{dataset_names[dataset]}}} & & \\nonumber \\\\\n")
                indices = np.argwhere(edge_acc_matrix > 0)
                edges_and_evidences = []
                already_done = set()
                for index in indices:
                    # if True or edge_count_matrix[index[0], index[1]] != 0 and edge_count_matrix[index[1], index[0]] != 0:
                    if edge_count_matrix[index[0], index[1]] + edge_count_matrix[index[1], index[0]] >= 3:
                        if (index[1], index[0]) in already_done:
                            continue
                        if edge_count_matrix[index[0], index[1]] > edge_count_matrix[index[1], index[0]]:
                            edges_and_evidences.append((index[0], index[1], edge_count_matrix[index[0], index[1]], edge_count_matrix[index[1], index[0]]))
                        else:
                            edges_and_evidences.append((index[1], index[0], edge_count_matrix[index[1], index[0]], edge_count_matrix[index[0], index[1]]))
                    already_done.add((index[0], index[1]))
                edges_and_evidences.sort(key=lambda x: x[2], reverse=True)
                # edges_and_evidences = edges_and_evidences[:5]
                for edge in edges_and_evidences:
                    # print(f"{unique_tags[edge[0]]} -> {unique_tags[edge[1]]} (evidence {edge[2]} vs {edge[3]})")
                    acc = int(100 * edge[2] / (edge[2] + edge[3]))
                    start = unique_tags[edge[0]].replace("_", "\\_")
                    end = unique_tags[edge[1]].replace("_", "\\_")
                    f.write(f"\\text{{``{start}''}} &~\\rightarrow~ \\text{{``{end}''}} && ({acc}\% ~/~ {edge[2] + edge[3]}~~) \\nonumber \\\\\n")

# write homogenity table to csv file
with open('plots/homogeneity/homogenity.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # write header
    header = ["LLM"] + datasets + ["Average"]
    writer.writerow(header)
    # write data
    for llm in llms:
        avg_accuracy = np.nanmean([accuracies[(llm, dataset)] for dataset in datasets])
        row = [llm] + [f"{accuracies[(llm, dataset)]:.4f}" for dataset in datasets] + [f"{avg_accuracy:.4f}"]
        writer.writerow(row)
    # average per dataset
    avg_accuracies = [sum([accuracies[(llm, dataset)] for llm in llms]) / len(llms) for dataset in datasets]
    writer.writerow(["Average"] + [f"{np.nanmean([accuracies[(llm, dataset)] for llm in llms]):.4f}" for dataset in datasets] + [f"{np.nanmean(avg_accuracies):.4f}"])
