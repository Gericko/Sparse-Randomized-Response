import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from math import ceil
import time

from sympy import false
from tqdm import trange
from pathlib import Path
import argparse

from graph import smaller_neighbors, load_wiki, load_gplus, extract_random_subgraph, down_degree
from compressed_randomized_response import CompressedVector, encode_vector, get_Q_RR_from_neutral


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


def publish_adjacency_list_crr(graph: nx.Graph, node, Q, privacy_budget, alpha, beta, seed) -> CompressedVector:
    n = graph.number_of_nodes()
    nb_blocks = max(1, ceil(beta * privacy_budget * down_degree(graph, node)))
    diffs = get_diff_list(graph, node)
    vect = encode_vector(diffs, Q, privacy_budget, alpha, seed, n, nb_blocks)
    return vect


def publish_edge_list_crr(graph: nx.Graph, Q, privacy_budget, alpha, beta, seed):
    n = graph.number_of_nodes()
    seeds = seed.spawn(n)
    edge_list = {}
    for i, node in enumerate(graph):
        edge_list[node] = publish_adjacency_list_crr(graph, node, Q, privacy_budget, alpha, beta, seeds[i])
    return edge_list


class CompressedAdjacencyList:
    def __init__(self, graph: nx.Graph, node, privacy_budget, alpha, beta, seed):
        self.node = node
        self.privacy_budget = privacy_budget
        self.seed = seed
        Q = get_Q_RR_from_neutral(0, [1], privacy_budget)
        self.publish_adjacency_list = publish_adjacency_list_crr(graph, node, Q, privacy_budget, alpha, beta, seed)

    def has_edge(self, i: int) -> bool:
        if i > self.node:
            raise ValueError("the index needs to be smaller than the index of the publishing node")
        return bool(self.publish_adjacency_list.decode(i))

    def edge_estimation(self, i: int) -> float:
        expe = np.exp(self.privacy_budget)
        return ((expe + 1) * self.has_edge(i) - 1) / (expe - 1)

    def upload_cost(self):
        return self.publish_adjacency_list.expected_communication_cost

    def huffman_cost(self):
        return self.publish_adjacency_list.communication_cost()


class GraphCRR:
    """Class implementing graph publication via Compressed Randomized Response"""

    def __init__(self, graph, privacy_budget, alpha, beta, seed):
        """
        Constructor for GraphCRR
        """
        seeds = seed.spawn(graph.number_of_nodes())
        self.published_edges = {
            node: CompressedAdjacencyList(graph, node, privacy_budget, alpha, beta, seeds[i])
            for i, node in enumerate(graph)
        }

    def has_edge(self, i, j):
        if i > j:
            i, j = j, i
        return self.published_edges[j].has_edge(i)

    def edge_estimation(self, i: int, j: int) -> float:
        if i > j:
            i, j = j, i
        return self.published_edges[j].edge_estimation(i)

    def upload_cost(self):
        return sum(adjacency.upload_cost() for adjacency in self.published_edges.values())

    def huffman_cost(self):
        return sum(adjacency.huffman_cost() for adjacency in self.published_edges.values())


class LazyGraphCRR:
    def __init__(self, graph, privacy_budget, alpha, beta, seed):
        self.graph = graph
        self.privacy_budget = privacy_budget
        self.alpha = alpha
        self.beta = beta
        self.seeds = seed.spawn(graph.number_of_nodes())
        self.has_been_published = {node: False for node in graph}
        self._all_published = False
        self._upload_cost = 0
        self._huffman_cost = 0

    def get_adjacency_list(self, node: int) -> CompressedAdjacencyList:
        if self.has_been_published[node]:
            raise ValueError("The adjacency list of node {} has already been published".format(node))
        self.has_been_published[node] = True
        compressed_adjacency_list = CompressedAdjacencyList(self.graph, node, self.privacy_budget, self.alpha, self.beta, self.seeds[node])
        self._upload_cost += compressed_adjacency_list.upload_cost()
        self._huffman_cost += compressed_adjacency_list.huffman_cost()
        return compressed_adjacency_list

    def is_fully_published(self):
        if not self._all_published:
            self._all_published = all(self.has_been_published.values())
        return self._all_published

    def upload_cost(self):
        if not self.is_fully_published():
                raise ValueError("The adjacency list has not been fully published")
        return self._upload_cost

    def huffman_cost(self):
        if not self.is_fully_published():
            raise ValueError("The adjacency list has not been fully published")
        return self._huffman_cost


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
        if param["graph_size"] < graph.number_of_nodes():
            extracted_graph = extract_random_subgraph(graph, param["graph_size"], rng)
        else:
            extracted_graph = graph
        node = rng.choice(list(extracted_graph.nodes()), 1)[0]
        start_time = time.time()
        vect = publish_adjacency_list_crr(extracted_graph, node, Q, epsilon, param["alpha"], param["beta"], seeds[i])
        execution_time = time.time() - start_time
        result_list.append(
            {
                **param,
                "degree": extracted_graph.degree(node),
                "down_degree": down_degree(extracted_graph, node),
                "non_private_upload_cost": vect.non_private_communication_cost,
                "expected_upload_cost": vect.expected_communication_cost,
                "huffman_upload_cost": vect.communication_cost(),
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
