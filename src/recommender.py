import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from math import ceil
import time

from sympy.abc import epsilon
from tqdm import tqdm
from pathlib import Path
import argparse

from compressed_randomized_response import encode_vector, get_Q_RR_from_neutral


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


def get_movie_correspondence_dict(df):
    correspondences = {}
    for index, row in df.iterrows():
        correspondences[row["movieId"]] = index
    return correspondences


def preprocess_ratings(rating_df, correspondences, only_viewing=False):
    rating_df["movieId"] = rating_df["movieId"].map(correspondences)
    if only_viewing:
        make_diffs = lambda r: (int(r["movieId"]), 1, 0)
    else:
        make_diffs = lambda r: (int(r["movieId"]), r["rating"], 0.0)
    rating_df["comp_rating"] = rating_df.apply(make_diffs, axis=1)
    return rating_df.groupby("userId")["comp_rating"].apply(list).to_frame("ratings")


def get_ratings(only_viewing=False):
    movie_df = pd.read_csv(DATA_DIR / MOVIE_FILE)
    correspondences = get_movie_correspondence_dict(movie_df)
    nb_movies = len(correspondences)

    rating_df = pd.read_csv(DATA_DIR / RATING_FILE)
    rating_by_user = preprocess_ratings(rating_df, correspondences, only_viewing)

    return rating_by_user, nb_movies


def get_ratings_small(only_viewing=False):
    movie_df = pd.read_csv(
        DATA_DIR / MOVIE_FILE_SMALL,
        delimiter="|",
        encoding="ISO-8859-1",
        names=["movieId", "title", "release date", "video release date",
            "IMDb URL", "unknown", "Action", "Adventure", "Animation",
            "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
            "Thriller", "War", "Western"]
    )
    correspondences = get_movie_correspondence_dict(movie_df)
    nb_movies = len(correspondences)

    rating_df = pd.read_csv(DATA_DIR / RATING_FILE_SMALL, delimiter="\t", names=["userId", "movieId", "rating", "timestamp"])
    rating_by_user = preprocess_ratings(rating_df, correspondences, only_viewing)

    return rating_by_user, nb_movies


def get_dataset(name, only_viewing=False):
    if name == "small":
        return get_ratings_small(only_viewing)
    elif name == "32M":
        return get_ratings(only_viewing)
    else:
        raise ValueError("Dataset name must be 'small' or '32M', not {}".format(name))


def experience_ratings(rating_by_user: pd.DataFrame, nb_movies, seed, rng, param):
    rng = np.random.default_rng(rng)
    seeds = seed.spawn(param["nb_iter"])
    epsilon = param["privacy_budget"] / param["alpha"] / 2
    Q = get_Q_RR_from_neutral(0.0, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], epsilon)
    result_list = []
    sampled = rating_by_user.sample(n=param["nb_iter"], replace=True, random_state=rng, ignore_index=True)
    for index, row in tqdm(sampled.iterrows(), total=sampled.shape[0]):
        nb_blocks = max(1, ceil(param["beta"] * epsilon * len(row["ratings"])))
        start_time = time.time()
        vect = encode_vector(row["ratings"], Q, epsilon, param["alpha"], seeds[index], nb_movies, nb_blocks)
        execution_time = time.time() - start_time
        result_list.append(
            {
                **param,
                "nb_ratings": len(row["ratings"]),
                "upload_cost": vect.expected_communication_cost,
                "execution_time": execution_time,
            }
        )
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(
        DIR_LOGS
        / "recommender_{exp_name}_n{nb_iter}_e{privacy_budget}_"
        "a{alpha}_{date}.csv".format(**param, date=time.time()),
        index=False,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="experience on compressed ratings"
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
    experience_ratings(rating_by_user, nb_movies, seed, rng, config)
