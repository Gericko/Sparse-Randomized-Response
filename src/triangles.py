from itertools import combinations, islice
import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from math import ceil
from scipy.stats import laplace
import time
from tqdm import trange
from pathlib import Path
import argparse

from graph import smaller_neighbors, load_wiki, load_gplus, extract_random_subgraph
from compressed_graph import GraphCRR


MARGIN_DEGREES = 150
DEGREE_SHARE = 0.1
EDGE_SHARE = 0.45
TRIANGLE_SHARE = 0.45
BETA = 2.0
ENTROPY = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_LOGS = BASE_DIR / "logs"


def count_triangles_from_compressed_graph(graph: nx.Graph, compressed_graph: GraphCRR, privacy_budget: float, degrees: dict[int,float]):
    rv = laplace(0, 1 / privacy_budget)
    count = 0
    noise = 0
    for node in graph:
        threshold = max(ceil(degrees[node] + MARGIN_DEGREES), 0)
        clipped_neighbors = sorted(islice(smaller_neighbors(graph, node), threshold))
        for i, j in combinations(clipped_neighbors, 2):
            count += compressed_graph.edge_estimation(i, j)
        noise += (np.exp(privacy_budget) + 1) / (np.exp(privacy_budget) - 1) * threshold * rv.rvs()
    return count, noise


def estimate_triangles(graph: nx.Graph, privacy_budget: float, alpha: float, beta: float, seed: SeedSequence):
    degrees = {n: laplace(d, 1 / (DEGREE_SHARE * privacy_budget)).rvs() for n, d in graph.degree()}
    compressed_budget = EDGE_SHARE * privacy_budget / alpha / 2
    compressed_graph = GraphCRR(graph, compressed_budget, alpha, beta, seed)
    download_cost = compressed_graph.upload_cost()
    count, noise = count_triangles_from_compressed_graph(graph, compressed_graph, TRIANGLE_SHARE * privacy_budget, degrees)
    return count, noise, download_cost


def get_parser():
    parser = argparse.ArgumentParser(
        description="estimate the number of triangles in a graph"
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
    parser.add_argument("-i", "--nb_iter", type=int, default=1, help="number of runs")
    return parser


def get_graph(graph_name):
    if graph_name == "gplus":
        return load_gplus()
    elif graph_name == "wiki":
        return load_wiki()
    else:
        raise ValueError("Graph {} is unknown.".format(graph_name))


def experience_triangle(graph, seed, rng, param):
    rng = np.random.default_rng(rng)
    for _ in trange(param["nb_iter"]):
        extracted_graph = extract_random_subgraph(graph, param["graph_size"], rng)
        true_triangle = sum(nx.triangles(extracted_graph).values()) / 3
        start_time = time.time()
        count, noise, d_cost = estimate_triangles(extracted_graph, param["privacy_budget"], param["alpha"], param["beta"], seed)
        result = pd.DataFrame(
            [
                {
                    **param,
                    "true_count": true_triangle,
                    "count": count,
                    "noise": noise,
                    "download_cost": d_cost,
                    "execution_time": time.time() - start_time,
                }
            ]
        )
        result.to_csv(
            DIR_LOGS
            / "triangles_{exp_name}_g{graph}_n{graph_size}_e{privacy_budget}_"
            "a{alpha}_{date}.csv".format(**param, date=time.time()),
            index=False,
        )


if __name__ == "__main__":
    seed = SeedSequence(ENTROPY)
    rng = np.random.default_rng(seed)
    config = vars(get_parser().parse_args())
    graph = get_graph(config["graph"])
    experience_triangle(graph, seed, rng, config)
