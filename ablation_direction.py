import argparse
import itertools
import math
import os
import scipy

import networkx as nx
import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import Meek

from discovery.pc_tagged import pc_tagged
from util import load_data, load_from_llm, make_deterministic, skeleton_to_cg_graph, introduce_tag_errors

datasets = [
    "bnlearn_child",
    "bnlearn_earthquake",
    "bnlearn_insurance",
    "bnlearn_survey",
    "bnlearn_asia",
    "bnlearn_cancer",
    "bnlearn_alarm",
    "lucas",
    "bnlearn_hailfinder",
    "bnlearn_hepar2",
    "bnlearn_win95pts"
]
llms = [
    "Llama-3.3-70B-Instruct",
    "claude-3-5-sonnet-20241022",
    "gpt-4-0613",
    "gpt-4o-2024-08-06",
    "Qwen2.5-72B-Instruct",
]
# llms = ["gpt-4-0613"]

anti_tags = False
remove_duplicates = True
remove_singular_tags = False

min_samples = 1
min_prob_threshold = 0.5
prior_on_weight = False
always_meeks = True
redirect_existing_edges = False
redirecting_strategy = 0
min_prob_redirecting = 0.6
include_current_edge_as_evidence = True
include_redirected_edges_in_edge_count = True

# ablation: undirect 1 (or 2, 3, ...) edges and calculate accuracy over whether these edges are directed correctly

parser = argparse.ArgumentParser(description="Ablation study on edge direction.")
parser.add_argument(
    "--type",
    choices=["undirect", "remove", "inverse", "tags"],
    default="tags",
    help="Type of ablation to perform: undirect, remove, or inverse.",
)
parser.add_argument(
    "--param", type=int, default=1, help="Parameter for how many edges to ablate."
)
parser.add_argument(
    "--llm", type=str, choices=llms, default="all", help="LLM to use for tagging."
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for reproducibility."
)
parser.add_argument(
    "--error_rate", type=float, default=0.1, help="Error rate for introducing tag errors."
)
args = parser.parse_args()

ablation_type = args.type
comb_nr = args.param
which_llms = args.llm
seed = args.seed
error_rate = args.error_rate

make_deterministic(seed)

# if ablation_type == "tags":
#     assert comb_nr == 1  # this is how I use it, but the code should also work fine with larger comb_nr

if which_llms != "all":
    llms = [which_llms]

all_reports = {llm: {dataset: None for dataset in datasets} for llm in llms}
for llm in llms:
    for dataset in datasets:
        # count what happened per edge
        report = {
            "meeks_did_something": 0,
            "tagging_correct": 0,
            "tagging_incorrect": 0,
            "tagging_nothing": 0,
        }
        confidences = {"correct": [], "incorrect": []}

        # load data (graph)
        variables, var_labels, _, edges, positions, data = load_data(
            dataset, order_data="random", seed=seed
        )

        # load tags
        data_id = dataset.split("_")[-1] if dataset.startswith("bnlearn") else dataset
        tags = load_from_llm(
            f"{llm}_tag_{data_id}_True.txt",
            variables=var_labels,
            anti_tags=anti_tags,
            remove_duplicates=remove_duplicates,
            remove_singular_tags=remove_singular_tags,
        )
        tag_list = [tags[var] for var in var_labels]
        if ablation_type == "tags":
            tag_list = introduce_tag_errors(tags, tag_list, var_labels, error_rate=error_rate)

        adj = np.zeros((len(variables), len(variables)))
        for edge in edges:
            adj[variables.index(edge[0]), variables.index(edge[1])] = 1

        # true nx graph
        true_graph_nx = nx.DiGraph(adj)

        # true cg graph
        true_graph = skeleton_to_cg_graph(true_graph_nx, undirected=False)
        cg_true = CausalGraph(len(variables), None)
        cg_true.G.graph = true_graph

        if comb_nr >= len(edges):
            print(f"Combination number {comb_nr} is as large or larger than the number of edges {len(edges)}")
            all_reports[llm][dataset] = (report, confidences)
            print(f"Done: {llm} {dataset}")
            continue
        number_combs = scipy.special.comb(len(edges), comb_nr)
        if ablation_type == "remove" or ablation_type == "inverse":
            number_combs *= len(edges) - comb_nr
        if number_combs > 20000:
            draw_nr = comb_nr
            if ablation_type == "remove" or ablation_type == "inverse":
                draw_nr += 1
            all_combs = []
            while len(all_combs) < 20000:
                all_combs.append(np.random.choice(len(edges), draw_nr, replace=False))
                all_combs[-1] = [edges[i] for i in all_combs[-1]]
            assert len(all_combs) == 20000
        else:
            all_combs = list(itertools.combinations(edges, comb_nr))
            if ablation_type == "remove" or ablation_type == "inverse":
                all_combs_temp = []
                for comb in all_combs:
                    for edge in edges:
                        if edge not in comb:
                            all_combs_temp.append(comb + (edge,))
                all_combs = all_combs_temp
            assert len(all_combs) == number_combs and len(all_combs) <= 20000

        for comb in all_combs:
            edges_copy = edges.copy()

            # remove edges
            if ablation_type == "remove":
                for edge in comb[:-1]:
                    edges_copy.remove(edge)

            # inverse edges
            if ablation_type == "inverse":
                for edge in comb[:-1]:
                    edges_copy.remove(edge)
                    edges_copy.append((edge[1], edge[0]))

            if ablation_type == "remove" or ablation_type == "inverse":
                to_undirect = [comb[-1]]
            else:
                to_undirect = comb

            # undirect edges
            for edge in to_undirect:
                # add the edge in the other direction to indicate undirectedness
                edges_copy.append((edge[1], edge[0]))

            adj = np.zeros((len(variables), len(variables)))
            for edge in edges_copy:
                adj[variables.index(edge[0]), variables.index(edge[1])] = 1

            # to nx digraph
            graph_nx = nx.DiGraph(adj)

            # to cg graph
            graph = skeleton_to_cg_graph(graph_nx, undirected=False)
            cg = CausalGraph(len(variables), None)
            cg.G.graph = graph

            # test whether meeks directs that already
            # np.sum can be used to get the number of undirected edges
            # no undirected edges gives a sum of 0, and every undirected edge adds -2
            if ablation_type == "undirect" or ablation_type == "tags":
                cg_new = Meek.meek(cg)
                if np.sum(cg_new.G.graph) != -2 * comb_nr:
                    # skip this, as it already gets directed by meeks
                    report["meeks_did_something"] += 1
                    continue
                assert np.array_equal(cg.G.graph, cg_new.G.graph)

            # if this is not skipped, run tagging
            always_meeks = ablation_type == "undirect" or ablation_type == "tags"  # only true for undirect ablation
            force_tagging = ablation_type in ["remove", "inverse"]  # for both remove and inverse ablation
            cg_res, info = pc_tagged(
                0,
                data,
                tag_list,
                min_samples=min_samples,
                min_prob_threshold=min_prob_threshold,
                prior_on_weight=prior_on_weight,
                always_meeks=always_meeks,
                redirect_existing_edges=redirect_existing_edges,
                redirecting_strategy=redirecting_strategy,
                min_prob_redirecting=min_prob_redirecting,
                include_current_edge_as_evidence=include_current_edge_as_evidence,
                include_redirected_edges_in_edge_count=include_redirected_edges_in_edge_count,
                cg=cg,
                force_tagging=force_tagging
            )

            correct = 0
            incorrect = 0
            undecided = 0
            for edge in to_undirect:
                start_idx = variables.index(edge[0])
                end_idx = variables.index(edge[1])
                if (
                    cg_res.G.graph[start_idx, end_idx] == -1
                    and cg_res.G.graph[end_idx, start_idx] == -1
                ):
                    undecided += 1
                elif (
                    cg_res.G.graph[start_idx, end_idx] == -1
                    and cg_res.G.graph[end_idx, start_idx] == 1
                ):
                    correct += 1
                    idx = [
                        i
                        for i, tup in enumerate(info)
                        if tup[0] == (start_idx, end_idx)
                        or tup[0] == (end_idx, start_idx)
                    ]
                    assert len(idx) <= 1
                    if len(idx) == 1:
                        idx = idx[0]
                        confidences["correct"].append(info[idx][1])
                elif (
                    cg_res.G.graph[start_idx, end_idx] == 1
                    and cg_res.G.graph[end_idx, start_idx] == -1
                ):
                    incorrect += 1
                    idx = [
                        i
                        for i, tup in enumerate(info)
                        if tup[0] == (start_idx, end_idx)
                        or tup[0] == (end_idx, start_idx)
                    ]
                    assert len(idx) <= 1
                    if len(idx) == 1:
                        idx = idx[0]
                        confidences["incorrect"].append(info[idx][1])
                else:
                    raise ValueError("This should not happen")
            if ablation_type == "undirect" or ablation_type == "tags":
                assert correct + incorrect + undecided == comb_nr
            else:
                assert correct + incorrect + undecided == 1

            report["tagging_nothing"] += undecided
            report["tagging_correct"] += correct
            report["tagging_incorrect"] += incorrect
        all_reports[llm][dataset] = (report, confidences)
        print(f"Done: {llm} {dataset}")

total_meeks_did_something = sum(
    report["meeks_did_something"]
    for llm_reports in all_reports.values()
    for report, confidences in llm_reports.values()
)
total_tagging_correct = sum(
    report["tagging_correct"]
    for llm_reports in all_reports.values()
    for report, confidences in llm_reports.values()
)
total_tagging_incorrect = sum(
    report["tagging_incorrect"]
    for llm_reports in all_reports.values()
    for report, confidences in llm_reports.values()
)
total_tagging_nothing = sum(
    report["tagging_nothing"]
    for llm_reports in all_reports.values()
    for report, confidences in llm_reports.values()
)

print(f"Total Meeks did something: {total_meeks_did_something}")
print(f"Total tagging correct: {total_tagging_correct}")
print(f"Total tagging incorrect: {total_tagging_incorrect}")
print(f"Total tagging nothing: {total_tagging_nothing}")

conf_correct_count = len(
    [
        conf
        for llm_reports in all_reports.values()
        for report, confidences in llm_reports.values()
        for conf in confidences["correct"]
    ]
)
if conf_correct_count != 0:
    average_confidence_correct = (
        sum(
            conf
            for llm_reports in all_reports.values()
            for report, confidences in llm_reports.values()
            for conf in confidences["correct"]
        )
        / conf_correct_count
    )
else:
    average_confidence_correct = "NaN"

conf_incorrect_count = len(
    [
        conf
        for llm_reports in all_reports.values()
        for report, confidences in llm_reports.values()
        for conf in confidences["incorrect"]
    ]
)
if conf_incorrect_count != 0:
    average_confidence_incorrect = (
        sum(
            conf
            for llm_reports in all_reports.values()
            for report, confidences in llm_reports.values()
            for conf in confidences["incorrect"]
        )
        / conf_incorrect_count
    )
else:
    average_confidence_incorrect = "NaN"

print(
    f"Average confidence correct ({conf_correct_count}): {average_confidence_correct}"
)
print(
    f"Average confidence incorrect ({conf_incorrect_count}): {average_confidence_incorrect}"
)

if ablation_type == "tags":
    save_path = f"results/ablation/{ablation_type}/{comb_nr}_{seed}_{error_rate}"
else:
    save_path = f"results/ablation/{ablation_type}/{comb_nr}_{seed}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# save all_reports as csv
for llm, llm_reports in all_reports.items():
    with open(f"{save_path}/{llm}.csv", "w") as f:
        f.write(
            "dataset,meeks_did_something,tagging_correct,tagging_incorrect,tagging_nothing\n"
        )
        for dataset, (report, confidences) in llm_reports.items():
            f.write(
                f"{dataset},{report['meeks_did_something']},{report['tagging_correct']},{report['tagging_incorrect']},{report['tagging_nothing']}\n"
            )

# save confidences as text file
for llm, llm_reports in all_reports.items():
    with open(f"{save_path}/{llm}_confidences.txt", "w") as f:
        for dataset, (report, confidences) in llm_reports.items():
            f.write(f"{dataset}\n")
            f.write("Correct: ")
            for conf in confidences["correct"]:
                f.write(f"{conf},")
            f.write("\n")
            f.write("Incorrect: ")
            for conf in confidences["incorrect"]:
                f.write(f"{conf},")
            f.write("\n")
            f.write("\n")
