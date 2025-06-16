import os
import itertools
import numpy as np
from scipy.stats import rankdata

# run_eval.sh
experimental_series = "final_eval"
identifier = ["bnlearn_child", "bnlearn_earthquake", "bnlearn_insurance", "bnlearn_survey", "bnlearn_asia", "bnlearn_cancer", "bnlearn_alarm", "lucas", "bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]
order_data = ["random"]
seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tagging_approach_id = [0]  # unused
llm = ["Llama-3.3-70B-Instruct", "claude-3-5-sonnet-20241022", "gpt-4-0613", "gpt-4o-2024-08-06", "Qwen2.5-72B-Instruct"]
# llm = ["Llama-3.3-70B-Instruct"]
nr_samples = [10000]
pc_indep_test = ["chisq"]
pc_alpha = [0.05]
min_samples = [1, 2]
min_prob_threshold = [0.5]
anti_tags = [False]
remove_duplicates = [True]
remove_singular_tags=[True, False]
prior_on_weight = [True, False]
always_meeks=[True, False]
redirect_existing_edges = [True, False]
redirecting_strategy = [0, 1]
min_prob_redirecting = [0.6]
include_current_edge_as_evidence = [True, False]
include_redirected_edges_in_edge_count = [True]
parameters = [experimental_series, identifier, order_data, seed, tagging_approach_id, llm, nr_samples, pc_indep_test, pc_alpha, min_samples, min_prob_threshold, anti_tags, remove_duplicates, remove_singular_tags, prior_on_weight, always_meeks, redirect_existing_edges, redirecting_strategy, min_prob_redirecting, include_current_edge_as_evidence, include_redirected_edges_in_edge_count]

all_methods = ["true", "skel", "skel_v", "pc", "skel_v_meeks", "ges", "typed_pc_naive", "typed_pc_maj", "tag_pc_1", "tag_pc_0", "tag_pc_0_on_skel_v", "tag_pc_0_on_ges"]
methods_to_consider = all_methods
paper_table = 0
if paper_table == 0:
    methods_to_consider = ["pc", "ges", "typed_pc_naive", "typed_pc_maj", "tag_pc_1", "tag_pc_0", "tag_pc_0_on_ges"]
elif paper_table == 1:
    methods_to_consider = ["skel_v_meeks", "tag_pc_0_on_skel_v", "tag_pc_0_on_ges"]
elif paper_table == 2:
    methods_to_consider = ["pc", "ges", "typed_prop_pc", "typed_prop_ges", "typed_pc_naive", "typed_pc_maj", "tag_pc_1", "tag_pc_0", "tag_pc_0_on_ges"]
else:
    raise ValueError("Invalid paper_table value")

method_names = {"true": "True Graph", "skel": "Skeleton", "skel_v": "Skeleton (V)", "pc": "PC", "skel_v_meeks": "GT CPDAG", "ges": "GES", "typed_pc_naive": "Typed-PC (Naive)", "typed_pc_maj": "Typed-PC (Maj.)", "tag_pc_1": "Tagged-PC (AntiV)", "tag_pc_0": "Tagged-PC", "tag_pc_0_on_skel_v": "Tagging on GT CPDAG", "tag_pc_0_on_ges": "Tagged-GES", "typed_prop_pc": "PC + t-Propagation", "typed_prop_ges": "GES + t-Propagation"}
dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                    "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                    "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}

def read_eval_csv(file):
    with open(file, "r") as file:
        csv_content = file.read()
        metrics = []
        models = []
        for line in csv_content.splitlines():
            model = line.split(",")[0]
            metric = line.split(",")[1:]
            metrics.append(metric)
            models.append(model)
        return np.array(metrics, dtype=float), models

def metrics_to_ranks(metrics):
    # metrics should be in the form (n_samples, n_metrics) where ranks are computed per metric
    # n_metrics = 7, where the first 4 are to be maximized and the last 3 to be minimized
    ranks = np.zeros_like(metrics)
    for i in range(metrics.shape[1]):
        if i % 7 < 4:
            ranks[:, i] = rankdata(metrics[:, i], method='min')
        else:
            ranks[:, i] = rankdata(-metrics[:, i], method='min')
    return ranks

def read_results(parameters):
    all_evals = {}
    path = f"results/{parameters[0]}"
    models = []
    already_had_redirect_with_this = []
    for parameter_combination in itertools.product(*parameters[1:]):
        before_redirect_params = parameter_combination[:16]
        identifier, order_data, seed, tagging_approach_id, llm, nr_samples, pc_indep_test, pc_alpha, min_samples, min_prob_threshold, anti_tags, remove_duplicates, remove_singular_tags, prior_on_weight, always_meeks, redirect_existing_edges, redirecting_strategy, min_prob_redirecting, include_current_edge_as_evidence, include_redirected_edges_in_edge_count = parameter_combination
        if before_redirect_params in already_had_redirect_with_this and redirect_existing_edges == False:
            continue
        elif redirect_existing_edges == False:
            already_had_redirect_with_this.append(before_redirect_params)
            # while the following paramters do not matter, these are the default values that determined where the results were saved
            redirecting_strategy = 1
            include_current_edge_as_evidence = False
            include_redirected_edges_in_edge_count = True

        result_path = f"{path}/{identifier}/{order_data}/{seed}/{pc_indep_test}_{pc_alpha}_{nr_samples}/{llm}/{anti_tags}_{remove_duplicates}_{remove_singular_tags}_{prior_on_weight}_{min_samples}_{min_prob_threshold}/{always_meeks}_{redirect_existing_edges}_{redirecting_strategy}_{include_current_edge_as_evidence}_{include_redirected_edges_in_edge_count}_{min_prob_redirecting}/_tagging_alg0"
        eval_file = f"{result_path}/eval.csv"
        eval_result, temp_models = read_eval_csv(eval_file)  # assuming models is the same for all considered files
        if models == []:
            models = temp_models
        else:
            assert models == temp_models
        # ranks = metrics_to_ranks(eval_result)
        # all_evals[parameter_combination] = (eval_result, ranks)
        all_evals[parameter_combination] = eval_result
        if parameter_combination[0] in ["bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]:
            assert np.all(all_evals[parameter_combination][:, 2:4] == 0)
            all_evals[parameter_combination][:, 2:4] = np.nan
    return all_evals, models

def filter_methods(all_evals, all_models, selected_methods):
    # filter out methods and only return actual evaluation values (not ranks)
    new_all_evals = {}
    for parameter_combination, eval_result in all_evals.items():
        new_eval_result = []
        for method in selected_methods:
            index = all_models.index(method)
            new_eval_result.append(eval_result[index])
        new_all_evals[parameter_combination] = np.array(new_eval_result)
    return new_all_evals

def flatten_datasets(all_evals):
    # flatten all datasets for the same configuration in the same order
    new_all_evals = {}
    for parameter_combination, eval_result in all_evals.items():
        dataset = parameter_combination[0]
        config = tuple(parameter_combination[1:])
        if new_all_evals.get(config) is None:
            new_all_evals[config] = []
        new_all_evals[config].append((dataset, eval_result))
    for config in new_all_evals:
        new_all_evals[config].sort(key=lambda x: x[0])
        new_all_evals[config] = np.concatenate([x[1] for x in new_all_evals[config]], axis=1)
    return new_all_evals

def average_parameters(all_evals, to_average, return_stds=False):
    # average over parameters
    new_all_evals = {}
    for parameter_combination, eval_result in all_evals.items():
        config = tuple([parameter_combination[i] for i in range(len(parameter_combination)) if i not in to_average])
        if new_all_evals.get(config) is None:
            new_all_evals[config] = []
        new_all_evals[config].append(eval_result)
    all_evals_means = {}
    for config in new_all_evals:
        all_evals_means[config] = np.average(new_all_evals[config], axis=0)
    if return_stds:
        all_evals_stds = {}
        for config in new_all_evals:
            all_evals_stds[config] = np.std(new_all_evals[config], axis=0)
        return all_evals_means, all_evals_stds
    else:
        return all_evals_means

def get_best_config_by_f1(evals, method_index):
    # configs as list
    all_configs = list(evals.keys())
    # put configs into dictionary by dataset
    configs_per_dataset = {}
    for config in all_configs:
        conf_without_ds = config[1:]  # without dataset
        # -1 is the f1 score
        eval_score = evals[config][method_index, -1]
        if configs_per_dataset.get(conf_without_ds) is None:
            configs_per_dataset[conf_without_ds] = []
        configs_per_dataset[conf_without_ds].append(eval_score)
    # get average f1 score per config
    for conf_without_ds, scores in configs_per_dataset.items():
        configs_per_dataset[conf_without_ds] = np.average(scores)

    # print best 10 configs
    # method = methods_to_consider[method_index]
    # configs_list = [(conf, score) for conf, score in configs_per_dataset.items()]
    # sorted_configs_list = sorted(configs_list, key=lambda x: x[1])
    # print(f"Best 10 methods for {method}")
    # for conf, score in sorted_configs_list[-10:]:
    #     print_best_config(conf, method)  # enter the right method here
    #     print(score)
    # print(f"End of best 10 methods for {method}")

    # get best config
    best_configs = []
    best_score = 0
    for conf, score in configs_per_dataset.items():
        if best_score == score:
            best_configs.append(conf)
        elif score > best_score:
            best_configs = [conf]
            best_score = score

    print (f"Best f1 score: {best_score}")
    return best_configs
    

def print_metrics_nicely(metrics, methods):
    for i in range(metrics.shape[0]):
        method_string = (methods[i] + ":").ljust(16)
        metrics_string = ", ".join([f"{metrics[i, j]:.2f}" for j in range(metrics.shape[1])])
        print(f"{method_string} {metrics_string}")


def metrics_to_latex(file, metrics, methods, stds=None):
    # this method plots the average ranks over all datasets
    bf_values = np.zeros(metrics.shape[1])
    for i in range(metrics.shape[1]):
        bf_values[i] = np.min(metrics[:, i])
    with open(file, "w") as file:
        file.write("\\begin{tabular}{l|ccccccc}\n")
        file.write("& SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n")
        file.write(" & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks \\\\\n")
        file.write("\\hline\n")
        for i in range(metrics.shape[0]):
            method_str = method_names[methods[i]]
            file.write(f"{method_str} & ")
            for j in range(metrics.shape[1]):
                metric_str = "$"
                if metrics[i, j] == bf_values[j]:
                    metric_str += f"\\mathbf{{{metrics[i, j]:.2f}}}"
                else:
                    metric_str += f"{metrics[i, j]:.2f}"
                if stds is not None:
                    metric_str += f" {{\\scriptstyle \\pm {stds[i, j]:.2f}}}"
                metric_str += "$"
                if j == metrics.shape[1] - 1:
                    file.write(f"{metric_str} \\\\\n")
                else:
                    file.write(f"{metric_str} & ")
            if methods[i] == "ges" or methods[i] == "typed_pc_maj":
                file.write("\\hline\n")
        file.write("\\end{tabular}")


def metrics_to_latex_plus(file, data, datasets, methods, stds=None):
    # this methods plots all datasets individually
    with open(file, "w") as file:
        file.write("\\begin{tabular}{l|ccccccc}\n")
        file.write("\\textbf{Evaluation Results} & SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n")
        for dataset in datasets:
            file.write("\\hline \\hline\n")
            dataset_name = dataset_names[dataset]
            file.write(f"Dataset {dataset_name} & & & & & & & \\\\\n")
            file.write("\\hline\n")
            current_data = data[dataset]
            current_stds = stds[dataset] if stds is not None else None
            bf_values = np.zeros(current_data.shape[1])
            for i in range(current_data.shape[1]):
                if i < 4:
                    bf_values[i] = np.min(current_data[:, i])
                else:
                    bf_values[i] = np.max(current_data[:, i])
            for i in range(current_data.shape[0]):
                method_str = method_names[methods[i]]
                file.write(f"{method_str} & ")
                for j in range(current_data.shape[1]):
                    metric_str = "$"
                    value = current_data[i, j]
                    if np.isnan(value):
                        metric_str = "-"
                    elif value == bf_values[j]:
                        metric_str += f"\\mathbf{{{value:.2f}}}"
                    else:
                        metric_str += f"{value:.2f}"
                    if stds is not None and not np.isnan(value):
                        metric_str += f" {{\\scriptstyle \\pm {current_stds[i, j]:.2f}}}"
                    if not np.isnan(value):
                        metric_str += "$"
                    if j == current_data.shape[1] - 1:
                        file.write(f"{metric_str} \\\\\n")
                    else:
                        file.write(f"{metric_str} & ") 
        file.write("\\end{tabular}")


def check_best_configs_for_uniqueness(eval_data, idents, midx, best_confs):
    if len(best_confs) == 1:
        best_conf = best_confs[0]
    else:
        dataset_evals_temp = {ds : [] for ds in idents}
        for best_conf in best_confs:
            for eval in eval_data:
                if eval[1:] == best_conf:
                    dataset_evals_temp[eval[0]].append(eval_data[eval][midx])
        for ds in dataset_evals_temp:
            assert len(dataset_evals_temp[ds]) == len(best_confs)
            first_res = dataset_evals_temp[ds][0]
            for res in dataset_evals_temp[ds][1:]:
                assert np.allclose(first_res, res, equal_nan=True)  # if this turns out false, we have to think of what we want to do then
        best_conf = best_confs[0]
    return best_conf


def print_best_config(best_conf, m):
    print(f"Best config for {m}")
    order_data, tagging_approach_id, llm, nr_samples, pc_indep_test, pc_alpha, min_samples, min_prob_threshold, anti_tags, remove_duplicates, remove_singular_tags, prior_on_weight, always_meeks, redirect_existing_edges, redirecting_strategy, min_prob_redirecting, include_current_edge_as_evidence, include_redirected_edges_in_edge_count = best_conf
    print(f"{best_conf[2]}, min_samples: {min_samples}, remove_sing_tags: {remove_singular_tags}, prior: {prior_on_weight}, always_meeks: {always_meeks}, redirect_edges: {redirect_existing_edges}, strategy: {redirecting_strategy}, include_cur: {include_current_edge_as_evidence}, include_red: {include_redirected_edges_in_edge_count}")


all_evals_original, models = read_results(parameters)
save_csv = False
if save_csv:
    # save all_evals to csv
    with open(f"results/{experimental_series}/all_evals.csv", "w") as file:
        # coulmns
        parameters = [experimental_series, identifier, order_data, seed, tagging_approach_id, llm, nr_samples, pc_indep_test, pc_alpha, min_samples, min_prob_threshold, anti_tags, remove_duplicates, remove_singular_tags, prior_on_weight, always_meeks, redirect_existing_edges, redirecting_strategy, min_prob_redirecting, include_current_edge_as_evidence, include_redirected_edges_in_edge_count]
        file.write("dataset,order_data,seed,tagging_approach_id,llm,nr_samples,pc_indep_test,pc_alpha,min_samples,min_prob_threshold,anti_tags,remove_duplicates,remove_singular_tags, prior_on_weight, always_meeks,redirect_existing_edges,redirecting_strategy,min_prob_redirecting,include_current_edge_as_evidence,include_redirected_edges_in_edge_count,method,SHD,SHD(double_for_anticausal),SID_min,SID_max,precision,recall,F1\n")
        for key, value in all_evals_original.items():
            line_base = ",".join([str(x) for x in key])
            for idx, model in enumerate(models):
                line_str = line_base
                line_str += f",{model},{','.join(map(str, value[idx]))}\n"
                file.write(line_str)

all_evals_and_seeds = filter_methods(all_evals_original, models, methods_to_consider)
all_evals, all_evals_stds = average_parameters(all_evals_and_seeds, [2], return_stds=True)  # indices are identifier, data order, seed, ...

if True:
    # get the best config by choosing the one with the best f1 score average (across seeds and datasets)
    dataset_evals = {ds : [] for ds in identifier}
    best_config_by_method = {method : None for method in methods_to_consider}
    for method in methods_to_consider:
        method_idx = methods_to_consider.index(method)
        best_configs = get_best_config_by_f1(all_evals, method_idx)
        best_config = check_best_configs_for_uniqueness(all_evals, identifier, method_idx, best_configs)
        best_config_by_method[method] = best_config
        for eval in all_evals:
            if eval[1:] == best_config:
                dataset_evals[eval[0]].append(all_evals[eval][method_idx])
        if method == "typed_pc_naive" or method == "typed_pc_maj" or method == "tag_pc_0" or method == "tag_pc_0_on_ges":
            print_best_config(best_config, method)

    if paper_table == 1:
        methods_to_consider = methods_to_consider[:-1]  # we can remove tag pc, just needed to find the config for it
        best_config_by_method["tag_pc_0_on_skel_v"] = best_config_by_method["tag_pc_0_on_ges"]  # use the config from ges

    # ranks per seed
    ranks_per_seed = []
    average_rank_tables = []
    # consider seeds separately now
    for s in seed:
        rank_tables_current_seed = []
        dataset_evals = {ds : [] for ds in identifier}
        # collect the data for all seeds using their respective configs
        for method in methods_to_consider:
            method_idx = methods_to_consider.index(method)
            config = best_config_by_method[method]
            for eval in all_evals_and_seeds:
                # this weird thing checks that the config (conf) matches the eval config
                # conf does not contain a seed, but eval has it at index 2, so this needs to match as well
                if config[0] == eval[1] and config[1:] == eval[3:] and eval[2] == s:
                    dataset_evals[eval[0]].append(all_evals_and_seeds[eval][method_idx])
        for dataset in dataset_evals:
            assert len(dataset_evals[dataset]) == len(methods_to_consider)
            dataset_evals[dataset] = np.array(dataset_evals[dataset])
            rank_tables_current_seed.append(metrics_to_ranks(dataset_evals[dataset]))
        average_rank_tables.append(np.nanmean(np.array(rank_tables_current_seed), axis=0))
    all_average_ranks = np.average(np.array(average_rank_tables), axis=0)
    all_average_ranks_stds = np.std(np.array(average_rank_tables), axis=0)
    print("Average ranks overall:")
    print_metrics_nicely(all_average_ranks, methods_to_consider)
    if paper_table == 0:
        name = "all_ranks"
    elif paper_table == 1:
        name = "true_skeleton"
    elif paper_table == 2:
        name  = "all_ranks_plus"
    metrics_to_latex(f"results/{experimental_series}/{name}.txt", all_average_ranks, methods_to_consider, stds=all_average_ranks_stds)

    # now per dataset, no ranks
    data_for_table = {}
    data_for_table_stds = {}
    for dataset in identifier:
        for method in methods_to_consider:
            for eval in all_evals:
                if eval[0] == dataset and eval[1:] == best_config_by_method[method]:
                    data_for_table[(dataset, method)] = all_evals[eval][methods_to_consider.index(method)]
                    data_for_table_stds[(dataset, method)] = all_evals_stds[eval][methods_to_consider.index(method)]
    data_tables = {}
    data_tables_stds = {}
    for dataset in identifier:
        data_tables[dataset] = np.array([data_for_table[(dataset, method)] for method in methods_to_consider])
        data_tables_stds[dataset] = np.array([data_for_table_stds[(dataset, method)] for method in methods_to_consider])
    if paper_table == 0:
        name = "all_datasets"
        split = True
    elif paper_table == 1:
        name = "true_skeleton_datasets"
        split = False
    elif paper_table == 2:
        name = "all_datasets_plus"
        split = True
    if not split:
        datasets = ["bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", "bnlearn_asia", "lucas", "bnlearn_child", "bnlearn_alarm", "bnlearn_insurance", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"]
        metrics_to_latex_plus(f"results/{experimental_series}/{name}.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds)
    else:
        datasets = ["bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", "bnlearn_asia", "lucas"]
        metrics_to_latex_plus(f"results/{experimental_series}/{name}.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds)
        datasets = ["bnlearn_child", "bnlearn_alarm", "bnlearn_insurance", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"]
        metrics_to_latex_plus(f"results/{experimental_series}/{name}_2.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds)

print("Done")