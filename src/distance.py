import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from math import ceil
import time
from tqdm import trange
from pathlib import Path
import argparse

from compressed_randomized_response import encode_vector, get_Q_RR_from_neutral
from recommender import get_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DIR_LOGS = BASE_DIR / "logs"
FIG_DIR = BASE_DIR / "figures"
MOVIE_FILE = "movies.csv"
RATING_FILE = "ratings.csv"
MOVIE_FILE_SMALL = "u.item"
RATING_FILE_SMALL = "u.data"

ENTROPY = 42
BETA = 2.0


def get_real_distance(u, v):
    movie_list_u = [movie for (movie, _, _) in u["ratings"]]
    movie_list_v = [movie for (movie, _, _) in v["ratings"]]
    return len(set(movie_list_u) & set(movie_list_v))


def estimate_distance(user, compressed, privacy_budget):
    movie_list = [movie for (movie, _, _) in user["ratings"]]
    count = 0
    expe = np.exp(privacy_budget)
    for movie in movie_list:
        count += ((expe + 1) * compressed.decode(movie) - 1) / (expe - 1)
    return count


def classic_rr(diffs, size, epsilon, rng):
    L = [0] * size
    for i, v, _ in diffs:
        L[i] = v
    for i, x in enumerate(L):
        if rng.random() < 1 / (1 + np.exp(epsilon)):
            L[i] = 1 - x
    return L


def estimate_distance_rr(user_1, user_2, privacy_budget, rng):
    rng = np.random.default_rng(rng)
    movie_list = [movie for (movie, _, _) in user_1["ratings"]]
    count = 0
    expe = np.exp(privacy_budget)
    for movie in movie_list:
        is_in_list = int(movie in user_2["ratings"])
        is_flipped = int(rng.random() < 1 / (1 + expe))
        count += abs(is_in_list - is_flipped)
    return ((expe + 1) * count - len(movie_list)) / (expe - 1)


def experience_distance(rating_by_user: pd.DataFrame, nb_movies, seed, rng, param):
    rng = np.random.default_rng(rng)
    seeds = seed.spawn(param["nb_iter"])
    epsilon = param["privacy_budget"] / param["alpha"] / 2
    Q = get_Q_RR_from_neutral(0, [1], epsilon)
    result_list = []
    for i in trange(param["nb_iter"]):
        sampled = rating_by_user.sample(
            n=2, replace=False, random_state=rng, ignore_index=False
        )
        user_1 = sampled.loc[sampled.index[0]]
        user_2 = sampled.loc[sampled.index[1]]
        nb_blocks = max(1, ceil(param["beta"] * epsilon * len(user_2["ratings"])))
        start_time = time.time()
        vect = encode_vector(
            user_2["ratings"],
            Q,
            epsilon,
            param["alpha"],
            seeds[i],
            nb_movies,
            nb_blocks,
        )
        real_distance = get_real_distance(user_1, user_2)
        estimated_distance = estimate_distance(user_1, vect, epsilon)
        execution_time = time.time() - start_time
        rr_distance = estimate_distance_rr(user_1, user_2, param["privacy_budget"], rng)
        result_list.append(
            {
                **param,
                "nb_ratings_1": len(user_1["ratings"]),
                "nb_ratings_2": len(user_2["ratings"]),
                "real_distance": real_distance,
                "estimated_distance": estimated_distance,
                "estimated_distance_rr": rr_distance,
                "execution_time": execution_time,
            }
        )
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(
        DIR_LOGS / "distance_{exp_name}_d{dataset}_i{nb_iter}_e{privacy_budget}_"
        "a{alpha}_{date}.csv".format(**param, date=time.time()),
        index=False,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="experience on nearest neighbor estimation for compressed rating data"
    )
    parser.add_argument(
        "-o", "--exp_name", type=str, default="test", help="name of the experiment"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="small",
        choices=["small", "32M"],
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


if __name__ == "__main__":
    config = vars(get_parser().parse_args())
    seed = SeedSequence(config["entropy"])
    rng = np.random.default_rng(seed)
    rating_by_user, nb_movies = get_dataset(config["dataset"])
    experience_distance(rating_by_user, nb_movies, seed, rng, config)
