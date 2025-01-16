from typing import Any, Callable
from math import ceil
import numpy as np
import sympy
from numpy.random.bit_generator import SeedSequence
import time
import pandas as pd
import seaborn as sns
from tqdm import trange

from counter_based_prng import CounterGenerator
from poisson_private_representation import encode_ppr


def huffman_cost(k: int) -> float:
    return np.log2(k) + np.log2(1 + np.log2(k)) + 2


class _CompressedBlock:
    def __init__(self, Q, public_gen, k, size):
        self.Q = Q
        self.public_gen = public_gen
        self.k = k
        self.size = size

    def decode(self, i: int, x_i: int):
        if x_i < 0 or x_i >= self.size:
            raise IndexError("x_i should be between 0 and {}, got {}.".format(self.size, x_i))

        gen = self.public_gen.get_generator(self.size * self.k + x_i)
        return self.Q(gen, i)

    def communication_cost(self):
        return huffman_cost(self.k)


def _encode_block(diff_from_ref: list[tuple[int, Any, Any]], Q, epsilon: float, alpha: float, seed: SeedSequence, size: int):
    r_bd = np.exp(len(diff_from_ref) * epsilon)
    [seed_1, seed_2] = seed.spawn(2)
    private_gen = np.random.default_rng(seed_1)
    public_gen = CounterGenerator(seed_2)

    def r(k: int) -> float:
        lprob = 0
        for (x_i, v_i, c_i) in diff_from_ref:
            B = Q(public_gen.get_generator(size * k + x_i), x_i)
            lprob += bool(B == v_i) - bool(B == c_i)
        return np.exp(lprob * epsilon)

    k = encode_ppr(r, r_bd, private_gen, alpha)
    return _CompressedBlock(Q, public_gen, k, size)


class CompressedVector:
    def __init__(self, compressed_blocks: list[_CompressedBlock], block_size: int, permutation: Callable[[int], int], expected_cost:float):
        self.compressed_blocks = compressed_blocks
        self.block_size = block_size
        self.permutation = permutation
        self.expected_communication_cost = expected_cost

    def decode(self, index: int):
        q, r = divmod(self.permutation(index), self.block_size)
        return self.compressed_blocks[q].decode(index, r)

    def communication_cost(self):
        return sum(block.communication_cost() for block in self.compressed_blocks)


def get_permutation(size: int, seed: SeedSequence):
    if not sympy.isprime(size):
        raise ValueError('size must be prime')

    rng = np.random.default_rng(seed=seed)
    multiplier = rng.integers(1, size)
    addition_constant = rng.integers(size)

    def permutation(index: int) -> int:
        return (multiplier * index + addition_constant) % size

    return permutation


def encode_vector(diff_from_ref: list[tuple], Q, epsilon: float, alpha: float, seed: SeedSequence, size: int, nb_blocks: int):
    size = sympy.nextprime(size)
    block_size = ceil(size / nb_blocks)
    blocks: list[list[tuple[int, Any, Any]]] = [ [] for _ in range(nb_blocks) ]
    seeds = seed.spawn(nb_blocks + 1)
    permutation = get_permutation(size, seeds[nb_blocks])

    for (x_i, v_i, c_i) in diff_from_ref:
        q, r = divmod(permutation(x_i), block_size)
        blocks[q].append((r, v_i, c_i))

    compressed_blocks = [_encode_block(block, Q, epsilon, alpha, seeds[i], block_size) for i, block in enumerate(blocks)]

    b = min(1.0, (alpha - 1) / 2)
    expected_cost = (1 + np.log(2)) * epsilon * len(diff_from_ref) + ((1 + np.log(2)) * np.log2(3.56) / b + 2) * nb_blocks

    return CompressedVector(compressed_blocks, block_size, permutation, expected_cost)


def get_Q_RR_from_reference(reference, choices, epsilon):
    if len(choices) > len(set(choices)):
        raise ValueError('choices must be unique')

    if not set(reference) <= set(choices):
        raise ValueError('all elements of the reference must be in choices')

    npe = np.exp(epsilon)
    proba = 1 / (npe + len(choices) - 1)

    def Q(gen: np.random.Generator, index):
        choice = reference[index]
        p = [npe * proba if c == choice else proba for c in choices]
        return gen.choice(choices, p=p)

    return Q


def get_Q_RR_from_neutral(neutral, alternative_choices, epsilon):
    choices = [neutral] + alternative_choices
    if len(choices) > len(set(choices)):
        raise ValueError('choices must be unique')

    npe = np.exp(epsilon)
    proba = 1 / (npe + len(choices) - 1)

    def Q(gen: np.random.Generator, _):
        p = [npe * proba] + [proba] * len(alternative_choices)
        return gen.choice(choices, p=p)

    return Q


if __name__ == '__main__':
    reps = 100
    epsilon = 1
    alpha = 2
    seed = np.random.SeedSequence(42)
    Q = get_Q_RR_from_neutral(0, [1], epsilon)

    # (s1, s2, s3) = (0, 0, 0)
    # for _ in range(reps):
    #     vect = encode_vector([(12, 1, 0), (6, 1, 0)], Q, epsilon, 2, seed, 10000, 3000)
    #     s1 += vect.decode(72)
    #     s2 += vect.decode(12)
    #     s3 += vect.decode(6)
    #
    # print(s1/reps, s2/reps, s3/reps)

    def get_random_vector(size, degree, rng=None):
        rng = np.random.default_rng(rng)
        indexes = rng.choice(size, degree, replace=False)
        return [(i, 1, 0) for i in indexes]

    # degree_list = []
    # time_list = []
    #
    # for degree in range(0, 5):
    #     for _ in trange(reps):
    #         vect = get_random_vector(100, degree)
    #         start_time = time.time()
    #         block = _encode_block(vect, Q, epsilon, alpha, seed, 100)
    #         running_time = time.time() - start_time
    #         degree_list.append(degree)
    #         time_list.append(running_time)
    #
    # db = pd.DataFrame({'degree': degree_list, 'time': time_list})
    # sns.set_theme(style="darkgrid")
    # fig = sns.relplot(x="degree", y="time", data=db, kind="line")
    # fig.set_axis_labels("degree", "runtime (s)")
    # fig.savefig("../figures/variance_block_degree.png", dpi=300, bbox_inches="tight")


    budget_list = []
    time_list = []

    for epsilon in np.linspace(0.1, 1, 10):
        for _ in trange(reps):
            vect = get_random_vector(100, 3)
            start_time = time.time()
            block = _encode_block(vect, Q, epsilon, alpha, seed, 100)
            running_time = time.time() - start_time
            budget_list.append(epsilon)
            time_list.append(running_time)

    db = pd.DataFrame({'epsilon': budget_list, 'time': time_list})
    sns.set_theme(style="darkgrid")
    fig = sns.relplot(x="epsilon", y="time", data=db, kind="line")
    fig.set_axis_labels("epsilon", "runtime (s)")
    fig.savefig("../figures/variance_block_epsilon.png", dpi=300, bbox_inches="tight")
