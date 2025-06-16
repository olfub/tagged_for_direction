import argparse
import os
import networkx as nx

import util
from discovery.pc import pc
from causallearn.search.ScoreBased.GES import ges
from discovery.pc_tagged import pc_tagged
from discovery.pc_typed import tpc_majority_top1, tpc_naive, typed_meek, tmec_enumeration
from eval import evaluate


def process(graph, positions, labels, name_path, name, true_graph, adjacency_matrices, eval_scores):
    util.visualize_graph(
        graph,
        positions=positions,
        labels=labels,
        name=f"{name_path}/graph_{name}",
    )
    adjacency_matrices.append(nx.adjacency_matrix(graph).todense())
    skip_sid = "hepar2" in name_path or "win95pts" in name_path or "hailfinder" in name_path
    # skip_sid = False
    eval_scores.append([name] + evaluate(true_graph, graph, skip_sid=skip_sid))


def main():

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser(description="Run causal discovery with tagging.")
    parser.add_argument("--experimental_series", type=str, default="initial_testing", help="Experimental series identifier")
    parser.add_argument("--identifier", type=str, default="lucas", help="Dataset identifier")
    parser.add_argument("--order_data", type=str, choices=["default", "random", "invert"], default="random", help="Strategy for in which order to load the variables and data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tagging_approach_id", type=int, default=0, help="approach ID for tagging")
    # tagging_approach_id : 0 --> use own tags if possible, 1 --> use LLM
    parser.add_argument("--load_with_llm", type=str, default="Llama-3.3-70B-Instruct", help="LLM to load tags from")
    parser.add_argument("--nr_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--pc_indep_test", type=str, default="fisherz", help="Independence test for PC algorithm")
    parser.add_argument("--pc_alpha", type=float, default=0.05, help="Alpha value for PC algorithm")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum number of samples needed to consider directing")
    parser.add_argument("--min_prob_threshold", type=float, default=0.5, help="Probability threshold to consider directing (probability must be larger!)")
    parser.add_argument("--anti_tags", type=str2bool, default=False, help="Create and use anti-tags")
    parser.add_argument("--remove_duplicates", type=str2bool, default=True, help="Remove duplicate tags that contain the same set of variables")
    parser.add_argument("--remove_singular_tags", type=str2bool, default=True, help="Remove tags that contain only one variable")
    parser.add_argument("--prior_on_weight", type=str2bool, default=False, help="Apply a prior on tag weights")
    parser.add_argument("--always_meeks", type=str2bool, default=False, help="Always use Meek's rules for edge orientation")
    parser.add_argument("--redirect_existing_edges", type=str2bool, default=True, help="Also allow for redirecting of existing edges")
    parser.add_argument("--redirecting_strategy", type=int, default=1, help="Which strategy to apply when redirecting edges")
    parser.add_argument("--min_prob_redirecting", type=float, default=0.6, help="Probability threshold for redirecting edges (probability must be larger!)")
    parser.add_argument("--include_current_edge_as_evidence", type=str2bool, default=False, help="Include current edge as evidence when considering an edge redirection")
    parser.add_argument("--include_redirected_edges_in_edge_count", type=str2bool, default=True, help="Include redirected edges in edge count for directing edges later")

    args = parser.parse_args()

    experimental_series = args.experimental_series
    identifier = args.identifier
    order_data = args.order_data
    seed = args.seed
    own_tags = args.tagging_approach_id == 0
    llm_approach_id = args.tagging_approach_id
    load_with_llm = args.load_with_llm
    nr_samples = args.nr_samples
    pc_indep_test = args.pc_indep_test
    pc_alpha = args.pc_alpha
    min_samples = args.min_samples
    min_prob_threshold = args.min_prob_threshold
    anti_tags = args.anti_tags
    remove_duplicates = args.remove_duplicates
    remove_singular_tags = args.remove_singular_tags
    prior_on_weight = args.prior_on_weight
    always_meeks = args.always_meeks
    redirect_existing_edges = args.redirect_existing_edges
    redirecting_strategy = args.redirecting_strategy
    min_prob_redirecting = args.min_prob_redirecting
    include_current_edge_as_evidence = args.include_current_edge_as_evidence
    include_redirected_edges_in_edge_count = args.include_redirected_edges_in_edge_count

    # create path if not exists
    path = util.create_path(experimental_series, args)

    if load_with_llm == "":
        raise ValueError("Not using provided LLM tags is not correctly implemented anymore. Please provide a LLM to load tags from.")  # TODO 
        # pipeline, system_prompt = setup_llm()

    adjacency_matrices = []
    eval_scores = []

    # make deterministic
    util.make_deterministic(seed)

    # load dataset
    variables, var_labels, tags, edges, positions, samples = util.load_data(identifier, nr_samples, order_data=order_data, seed=seed)
    labels = {i: variables[i] for i in range(len(variables))}

    # create ANM object
    anm = util.create_model(variables, edges, seed)
    true_graph = anm.graph

    # sample from ANM
    if samples is None:
        samples = anm.sample(nr_samples)

    # visualize true graph
    pc_folder = f"{path}/_pc"
    if not os.path.exists(pc_folder):
        os.makedirs(pc_folder)
        process(true_graph, positions, labels, pc_folder, "true", true_graph, adjacency_matrices, eval_scores)

        # get and visualize ground truth skeleton
        skeleton, skeleton_with_v = util.get_skeleton(anm.graph)

        # visualize and evaluate skeleton
        process(skeleton, positions, labels, pc_folder, "skel", true_graph, adjacency_matrices, eval_scores)

        # visualize and evaluate skeleton with v-structures
        process(skeleton_with_v, positions, labels, pc_folder, "skel_v", true_graph, adjacency_matrices, eval_scores)

        # apply normal PC
        cg_pc, cg_1_pc = pc(samples, indep_test=pc_indep_test, alpha=pc_alpha)
        graph = util.cg_to_nx(cg_pc)
        pc_skeleton, pc_skeleton_with_v = util.get_skeleton(graph)
        pc_sepset = cg_pc.sepset.copy()
        process(graph, positions, labels, pc_folder, "pc", true_graph, adjacency_matrices, eval_scores)

        # apply PC on ground truth skeleton with v-structures (i.e., just apply meeks)
        cg_skel_v_meeks, _ = pc(samples, indep_test=pc_indep_test, alpha=pc_alpha, gt_skeleton=util.skeleton_to_cg_graph(skeleton_with_v, undirected=False))
        graph_skel_v_meeks = util.cg_to_nx(cg_skel_v_meeks)
        process(graph_skel_v_meeks, positions, labels, pc_folder, "skel_v_meeks", true_graph, adjacency_matrices, eval_scores)

        util.save_pc_res(pc_folder, cg_pc, cg_1_pc, skeleton_with_v, pc_skeleton, pc_sepset, adjacency_matrices, eval_scores)
    else:
        cg_pc, cg_1_pc, skeleton_with_v, pc_skeleton, pc_sepset, adjacency_matrices, eval_scores = util.load_pc_res(pc_folder)

    # ges
    ges_folder = f"{path}/_ges"
    if not os.path.exists(ges_folder):
        os.makedirs(ges_folder)

        # apply normal GES
        result = ges(samples, score_func="local_score_BDeu")
        ges_graph = result["G"]
        graph_ges = util.ges_to_nx(ges_graph)
        process(graph_ges, positions, labels, ges_folder, "ges", true_graph, adjacency_matrices, eval_scores)

        util.save_ges_res(ges_folder, graph_ges, adjacency_matrices, eval_scores)
    else:
        graph_ges, adjacency_matrices, eval_scores = util.load_ges_res(ges_folder)

    path_llm = f"{path}/{load_with_llm}"

    typing_folder = f"{path_llm}/_typing"
    if not os.path.exists(typing_folder):
        os.makedirs(typing_folder)
        # get types
        if load_with_llm != "":
            if identifier.startswith("bnlearn"):
                data_id = identifier.split("_")[1]
            else:
                data_id = identifier
            types = util.load_from_llm(f"{load_with_llm}_type_{data_id}_True.txt", variables=var_labels)
            type_list = list(set(types.values()))
        else:
            raise ValueError("Not using provided LLM tags is not correctly implemented anymore. Please provide a LLM to load tags from.")  # TODO 
            # types = get_types(approach_id=1, variables=var_labels, save_name=f"{path}/{id}", pipeline=pipeline, system_prompt=system_prompt)
            # type_list = list(set(types.values()))
        extra_type_index = len(type_list)
        for node, var in zip(pc_skeleton.nodes, var_labels):
            if var in types:
                pc_skeleton.nodes[node]["type"] = type_list.index(types[var])
            else:
                pc_skeleton.nodes[node]["type"] = extra_type_index
                extra_type_index += 1

        # typing naive
        graph = tpc_naive(pc_skeleton, util.prep_sep_sets_for_typing(pc_sepset))
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
        process(graph, positions, labels, typing_folder, "typed_pc_naive", true_graph, adjacency_matrices, eval_scores)
        
        # typing majority
        graph = tpc_majority_top1(pc_skeleton, util.prep_sep_sets_for_typing(pc_sepset))
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
        process(graph, positions, labels, typing_folder, "typed_pc_maj", true_graph, adjacency_matrices, eval_scores)

        # typing propagation
        extra_type_index = len(type_list)
        types_as_list = []
        for var in var_labels:
            if var in types:
                types_as_list.append(type_list.index(types[var]))
            else:
                types_as_list.append(extra_type_index)
                extra_type_index += 1

        adjacency_matrix = nx.adjacency_matrix(pc_skeleton).todense()
        pre_tess_g, g_compatibility = typed_meek(adjacency_matrix, types_as_list)
        # all_dags = tmec_enumeration(pre_tess_g, types_as_list, g_compatibility)
        # tess_g = (sum(all_dags) > 0).astype(int)
        tess_g = pre_tess_g
        graph = nx.from_numpy_array(tess_g, create_using=nx.DiGraph)
        process(graph, positions, labels, typing_folder, "typed_prop_pc", true_graph, adjacency_matrices, eval_scores)

        adjacency_matrix = nx.adjacency_matrix(graph_ges).todense()
        pre_tess_g, g_compatibility = typed_meek(adjacency_matrix, types_as_list)
        # all_dags = tmec_enumeration(pre_tess_g, types_as_list, g_compatibility)
        # tess_g = (sum(all_dags) > 0).astype(int)
        tess_g = pre_tess_g
        graph = nx.from_numpy_array(tess_g, create_using=nx.DiGraph)
        process(graph, positions, labels, typing_folder, "typed_prop_ges", true_graph, adjacency_matrices, eval_scores)

        util.save_adj_eval(typing_folder, adjacency_matrices, eval_scores)
    else:
        adjacency_matrices, eval_scores = util.load_adj_eval(typing_folder)

    # get tags
    if load_with_llm != "":
        if identifier.startswith("bnlearn"):
            data_id = identifier.split("_")[1]
        else:
            data_id = identifier
        tags = util.load_from_llm(f"{load_with_llm}_tag_{data_id}_True.txt", variables=var_labels, anti_tags=anti_tags, remove_duplicates=remove_duplicates, remove_singular_tags=remove_singular_tags)
        tag_list = [tags[var] for var in var_labels]
    else:
        raise ValueError("Not using provided LLM tags is not correctly implemented anymore. Please provide a LLM to load tags from.")  # TODO 

    path_tagging = f"{path_llm}/{anti_tags}_{remove_duplicates}_{remove_singular_tags}_{prior_on_weight}_{min_samples}_{min_prob_threshold}"
    tagging_alg1_folder = f"{path_tagging}/_tagging_alg1"
    if not os.path.exists(tagging_alg1_folder):
        os.makedirs(tagging_alg1_folder)
        # apply tagged PC, algorithm 1
        cg, info_tag_pc_1 = pc_tagged(1, samples, tag_list, indep_test=pc_indep_test, alpha=pc_alpha, min_samples=min_samples, min_prob_threshold=min_prob_threshold, prior_on_weight=prior_on_weight, 
                            always_meeks=always_meeks, redirect_existing_edges=redirect_existing_edges, redirecting_strategy=redirecting_strategy, min_prob_redirecting=min_prob_redirecting, 
                            include_current_edge_as_evidence=include_current_edge_as_evidence, include_redirected_edges_in_edge_count=include_redirected_edges_in_edge_count,
                            cg=cg_1_pc)
        graph = util.cg_to_nx(cg)
        process(graph, positions, labels, tagging_alg1_folder, "tag_pc_1", true_graph, adjacency_matrices, eval_scores)

        util.save_adj_eval(tagging_alg1_folder, adjacency_matrices, eval_scores)
        util.save_info(info_tag_pc_1, true_graph, f"{tagging_alg1_folder}/tag_pc_1", variables, use_pickle=True)
    else:
        adjacency_matrices, eval_scores = util.load_adj_eval(tagging_alg1_folder)
        info_tag_pc_1 = util.load_info_pickle(f"{tagging_alg1_folder}/tag_pc_1")

    path_tagging_alg0 = f"{path_tagging}/{always_meeks}_{redirect_existing_edges}_{redirecting_strategy}_{include_current_edge_as_evidence}_{include_redirected_edges_in_edge_count}_{min_prob_redirecting}"
    tagging_alg0_folder = f"{path_tagging_alg0}/_tagging_alg0"
    os.makedirs(tagging_alg0_folder, exist_ok=True)

    # apply tagged PC, algorithm 0
    cg, info_tag_pc_0 = pc_tagged(0, samples, tag_list, indep_test=pc_indep_test, alpha=pc_alpha, min_samples=min_samples, min_prob_threshold=min_prob_threshold, prior_on_weight=prior_on_weight, 
                         always_meeks=always_meeks, redirect_existing_edges=redirect_existing_edges, redirecting_strategy=redirecting_strategy, min_prob_redirecting=min_prob_redirecting, 
                         include_current_edge_as_evidence=include_current_edge_as_evidence, include_redirected_edges_in_edge_count=include_redirected_edges_in_edge_count, cg=cg_pc)
    graph = util.cg_to_nx(cg)
    process(graph, positions, labels, tagging_alg0_folder, "tag_pc_0", true_graph, adjacency_matrices, eval_scores)

    # apply tagged PC algorithm 0 on correct skeleton including v-structures
    gt_skeleton = util.skeleton_to_cg_graph(skeleton_with_v, undirected=False)
    cg, info_tag_pc_0_on_skel_v = pc_tagged(0, samples, tag_list, indep_test=pc_indep_test, alpha=pc_alpha, gt_skeleton=gt_skeleton, search_v_structures=False, min_samples=min_samples, min_prob_threshold=min_prob_threshold,
                            prior_on_weight=prior_on_weight, always_meeks=always_meeks, redirect_existing_edges=redirect_existing_edges, redirecting_strategy=redirecting_strategy, min_prob_redirecting=min_prob_redirecting, 
                            include_current_edge_as_evidence=include_current_edge_as_evidence, include_redirected_edges_in_edge_count=include_redirected_edges_in_edge_count, cg=cg_pc)
    graph = util.cg_to_nx(cg)
    process(graph, positions, labels, tagging_alg0_folder, "tag_pc_0_on_skel_v", true_graph, adjacency_matrices, eval_scores)

    # apply tagged (PC) algorithm 0 on ges graph (this doesn't really have anything to do with anymore but it still tests tagging)
    skeleton = util.skeleton_to_cg_graph(graph_ges, undirected=False)
    cg, info_tag_pc_0_on_ges = pc_tagged(0, samples, tag_list, indep_test=pc_indep_test, alpha=pc_alpha, gt_skeleton=skeleton, search_v_structures=False, min_samples=min_samples, min_prob_threshold=min_prob_threshold,
                            prior_on_weight=prior_on_weight, always_meeks=always_meeks, redirect_existing_edges=redirect_existing_edges, redirecting_strategy=redirecting_strategy, min_prob_redirecting=min_prob_redirecting, 
                            include_current_edge_as_evidence=include_current_edge_as_evidence, include_redirected_edges_in_edge_count=include_redirected_edges_in_edge_count, cg=cg_pc)
    graph = util.cg_to_nx(cg)
    process(graph, positions, labels, tagging_alg0_folder, "tag_pc_0_on_ges", true_graph, adjacency_matrices, eval_scores)

    infos = [info_tag_pc_0, info_tag_pc_0_on_skel_v, info_tag_pc_1, info_tag_pc_0_on_ges]
    info_names = ["tag_pc_0", "tag_pc_0_on_skel_v", "tag_pc_1", "tag_pc_0_on_ges"]
    util.save_all(name=tagging_alg0_folder, infos=infos, info_names=info_names, true_graph=anm.graph, eval_scores=eval_scores,
                  adjacency_matrices=adjacency_matrices, variables=variables, args=args)


if __name__ == "__main__":
    main()
