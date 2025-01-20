import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from math import ceil
import time
from tqdm import trange
from pathlib import Path
import argparse

from graph import smaller_neighbors, load_wiki, load_gplus, extract_random_subgraph, down_degree
from compressed_randomized_response import encode_vector, get_Q_RR_from_neutral


MARGIN_DEGREES = 150
BETA = 2.0
ENTROPY = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_LOGS = BASE_DIR / "logs"


def get_diff_list(graph, node):
    return [
        (v, 1, 0)
        for v in smaller_neighbors(graph, node)
    ]


def publish_adjacency_list_crr(graph: nx.Graph, node, Q, privacy_budget, alpha, beta, seed):
    n = graph.number_of_nodes()
    nb_blocks = max(1, ceil(beta * privacy_budget * down_degree(graph, node)))
    diffs = get_diff_list(graph, node)
    vect = encode_vector(diffs, Q, privacy_budget, alpha, seed, n, nb_blocks)
    return vect


def publish_edge_list_crr(graph: nx.Graph, Q, privacy_budget, alpha, beta, seed):
    n = graph.number_of_nodes()
    seeds = seed.spawn(n)
    edge_list = {}
    for i, node in enumerate(graph.nodes):
        edge_list[node] = publish_adjacency_list_crr(graph, node, Q, privacy_budget, alpha, beta, seeds[i])
    return edge_list


class GraphCRR:
    """Class implementing graph publication via Compressed Randomized Response"""

    def __init__(self, graph, privacy_budget, alpha, beta, seed):
        """
        Constructor for GraphCRR
        """
        self.privacy_budget = privacy_budget
        self.seed = seed
        Q = get_Q_RR_from_neutral(0, [1], privacy_budget)
        self.published_edges = publish_edge_list_crr(graph, Q, privacy_budget, alpha, beta, seed)

    def has_edge(self, i, j):
        if i > j:
            i, j = j, i
        return bool(self.published_edges[j].decode(i))

    def edge_estimation(self, i: int, j: int) -> float:
        expe = np.exp(self.privacy_budget)
        return ((expe + 1) * self.has_edge(i, j) - 1) / (expe - 1)

    def upload_cost(self):
        return sum(vect.expected_communication_cost for vect in self.published_edges.values())

    def huffman_cost(self):
        return sum(vect.communication_cost() for vect in self.published_edges.values())


def get_parser():
    parser = argparse.ArgumentParser(
        description="experience on compressed adjacency lists"
    )
    parser.add_argument(
        "-o", "--exp_name", type=str, default="test", help="name of the experiment"
    )
    parser.add_argument(
        "-g",
        "--graph",
        type=str,
        default="wiki",
        choices=["gplus", "wiki"],
    )
    parser.add_argument(
        "-n",
        "--graph_size",
        type=int,
        default=7115,
        help="size of the graph to extract",
    )
    parser.add_argument(
        "-e",
        "--privacy_budget",
        type=float,
        default=1,
        help="privacy budget of the algorithm",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=2,
        help="parameter alpha of the poisson private representation",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=BETA,
        help="parameter beta of the algorithm",
    )
    parser.add_argument("-s", "--entropy", type=int, default=ENTROPY)
    parser.add_argument("-i", "--nb_iter", type=int, default=1, help="number of runs")
    return parser


def get_graph(graph_name):
    if graph_name == "gplus":
        return load_gplus()
    elif graph_name == "wiki":
        return load_wiki()
    else:
        raise ValueError("Graph {} is unknown.".format(graph_name))


def experience_adjacency(graph, seed, rng, param):
    rng = np.random.default_rng(rng)
    seeds = seed.spawn(param["nb_iter"])
    epsilon = param["privacy_budget"] / param["alpha"] / 2
    Q = get_Q_RR_from_neutral(0, [1], epsilon)
    result_list = []
    for i in trange(param["nb_iter"]):
        extracted_graph = extract_random_subgraph(graph, param["graph_size"], rng)
        node = rng.choice(list(extracted_graph.nodes()), 1)[0]
        start_time = time.time()
        vect = publish_adjacency_list_crr(extracted_graph, node, Q, epsilon, param["alpha"], param["beta"], seeds[i])
        execution_time = time.time() - start_time
        result_list.append(
            {
                **param,
                "degree": extracted_graph.degree(node),
                "upload_cost": vect.expected_communication_cost,
                "execution_time": execution_time,
            }
        )
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(
            DIR_LOGS
            / "adjacency_{exp_name}_g{graph}_n{graph_size}_e{privacy_budget}_"
            "a{alpha}_{date}.csv".format(**param, date=time.time()),
            index=False,
        )


if __name__ == "__main__":
    config = vars(get_parser().parse_args())
    seed = SeedSequence(config["entropy"])
    rng = np.random.default_rng(seed)
    graph = get_graph(config["graph"])
    experience_adjacency(graph, seed, rng, config)
