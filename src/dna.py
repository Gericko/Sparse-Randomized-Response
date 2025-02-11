import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from numpy.random import SeedSequence
from math import ceil
import time
from tqdm import trange
from pathlib import Path
import argparse
import seaborn as sns

from compressed_randomized_response import encode_vector, get_Q_RR_from_neutral


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DIR_LOGS = BASE_DIR / "logs"
FIG_DIR = BASE_DIR / "figures"
DNA_FILE = "processed_snp_chr22.csv"

ENTROPY = 42
BETA = 2.0


def plot_probs_distribution(df, file_name="snps_proba_distribution.png"):
    sns.set_theme(style="darkgrid")
    fig = sns.displot(df, x="proba", kind="ecdf", log_scale=(True, False))
    fig.savefig(FIG_DIR / file_name, dpi=300, bbox_inches="tight")


def clip_frequency(df, threshold=0.01):
    return df[df["proba"] <= threshold]


def get_random_vector(frequencies, rng=None):
    rng = np.random.default_rng(rng)
    mask = bernoulli.rvs(p=frequencies, random_state=rng)
    indexes = mask.nonzero()[0]
    return [(i, 1, 0) for i in indexes]


def experience_dna(frequencies, seed, rng, param):
    rng = np.random.default_rng(rng)
    seeds = seed.spawn(param["nb_iter"])
    epsilon = param["privacy_budget"] / param["alpha"] / 2
    Q = get_Q_RR_from_neutral(0, [1], epsilon)
    result_list = []
    for i in trange(param["nb_iter"]):
        diffs = get_random_vector(frequencies, rng)
        nb_blocks = max(1, ceil(param["beta"] * epsilon * len(diffs)))
        start_time = time.time()
        vect = encode_vector(
            diffs, Q, epsilon, param["alpha"], seeds[i], len(frequencies), nb_blocks
        )
        execution_time = time.time() - start_time
        result_list.append(
            {
                **param,
                "nb_variations": len(diffs),
                "nb_locations": len(frequencies),
                "non_private_upload_cost": vect.non_private_communication_cost,
                "expected_upload_cost": vect.expected_communication_cost,
                "huffman_upload_cost": vect.communication_cost(),
                "execution_time": execution_time,
            }
        )
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(
        DIR_LOGS / "dna_{exp_name}_n{nb_iter}_e{privacy_budget}_"
        "a{alpha}_{date}.csv".format(**param, date=time.time()),
        index=False,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="experience on DNA SNPS")
    parser.add_argument(
        "-o", "--exp_name", type=str, default="test", help="name of the experiment"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.01,
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
    df = pd.read_csv(
        DATA_DIR / DNA_FILE, sep=" ", names=["chr", "pos", "ref", "alt", "proba"]
    )
    print("The original dataset contains {} variations".format(len(df)))
    df_clipped = clip_frequency(df, config["threshold"])
    percentage = len(df_clipped) / len(df) * 100
    print(
        "After clipped at {}, the dataset contains {} variations, ie. {:.2f}%".format(
            config["threshold"], len(df_clipped), percentage
        )
    )
    proba_vect = df_clipped["proba"].to_numpy()
    experience_dna(proba_vect, seed, rng, config)
