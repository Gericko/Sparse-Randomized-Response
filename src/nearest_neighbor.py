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


def get_correspondences_from_list(movie_list):
    return {movie: i for i, movie in enumerate(movie_list)}


def publish_sliced_ratings(ratings, nb_movies, correspondences, Q, privacy_budget, alpha, beta, seed):
    seeds = seed.spawn(ratings.shape[0])
    compressed_ratings = {}
    for index, row in ratings.iterrows():
        compressed_ratings[index] = row["ratings"]
        nb_blocks = max(1, ceil(beta * privacy_budget * len(row["ratings"])))
        diffs = [(correspondences[movie], x, y) for (movie, x, y) in row["ratings"] if movie in correspondences]
        vect = encode_vector(diffs, Q, privacy_budget, alpha, seeds[index], nb_movies, nb_blocks)
        compressed_ratings[index] = vect
    return compressed_ratings


class SliceCompressedRatings:
    def __init__(self, ratings, nb_movies, Q, user, privacy_budget, alpha, beta, seed):
        self.user = user
        self.users = list(ratings.index)
        self.movie_list = [movie for (movie, _, _) in ratings.loc[user, "ratings"]]
        self.Q = Q
        self.privacy_budget = privacy_budget
        self.seed = seed
        self.correspondences = get_correspondences_from_list(self.movie_list)
        self.compressed_ratings = publish_sliced_ratings(ratings, nb_movies, self.correspondences, Q, privacy_budget, alpha, beta, seed)

    def get_rating(self, user, movie):
        try:
            return self.compressed_ratings[user].decode(self.correspondences[movie])
        except KeyError:
            raise KeyError(f"Movie {movie} is not part of the movie list")

    def get_distance(self, user):
        return sum(self.get_rating(user, movie) for movie in self.movie_list)

    def get_distances(self):
        return {user: self.get_distance(user) for user in self.users}

    def get_estimated_nearest_neighbor(self):
        distances = self.get_distances()
        del distances[self.user]
        return max(distances, key=distances.get)


def get_distances_to_movie_list(ratings: pd.DataFrame, movie_list) -> pd.Series:
    movie_set = set(movie_list)
    return ratings.apply(lambda row: sum(1 for (movie, _, _) in row["ratings"] if movie in movie_set), axis=1)


def get_rank_of_estimation(ratings, sliced_ratings):
    ground_truth_distances = get_distances_to_movie_list(ratings, sliced_ratings.movie_list)
    estimated_nearest_neighbor = sliced_ratings.get_estimated_nearest_neighbor()
    return ground_truth_distances.rank(method="min", ascending=False)[estimated_nearest_neighbor]


def experience_nearest_neighbor(rating_by_user: pd.DataFrame, nb_movies, seed, rng, param):
    rng = np.random.default_rng(rng)
    seeds = seed.spawn(param["nb_iter"])
    epsilon = param["privacy_budget"] / param["alpha"] / 2
    Q = get_Q_RR_from_neutral(0, [1], epsilon)
    result_list = []
    for i in trange(param["nb_iter"]):
        sampled = rating_by_user.sample(n=param["size"], replace=False, random_state=rng, ignore_index=True)
        user = sampled.sample(n=1, random_state=rng, ignore_index=False).index[0]
        start_time = time.time()
        compressed = SliceCompressedRatings(sampled, nb_movies, Q, user, epsilon, param["alpha"], param["beta"], seeds[i])
        rank = get_rank_of_estimation(sampled, compressed)
        execution_time = time.time() - start_time
        result_list.append(
            {
                **param,
                "nb_movies": len(compressed.movie_list),
                "rank": rank,
                "execution_time": execution_time,
            }
        )
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(
            DIR_LOGS
            / "neighbor_{exp_name}_n{size}_e{privacy_budget}_"
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
        "-n",
        "--size",
        type=int,
        default=943,
        help="number of users to extract from the dataset",
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


if __name__ == "__main__":
    seed = SeedSequence(ENTROPY)
    rng = np.random.default_rng(seed)
    config = vars(get_parser().parse_args())
    rating_by_user, nb_movies = get_dataset(config["dataset"])
    experience_nearest_neighbor(rating_by_user, nb_movies, seed, rng, config)
