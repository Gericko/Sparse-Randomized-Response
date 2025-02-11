# This is a modified file from the source code of the following article:
#
# Liu, Yanxiao, Wei-Ning Chen, Ayfer Özgür, and Cheuk Ting Li.
# "Universal Exact Compression of Differentially Private Mechanisms."
# arXiv preprint arXiv:2405.20782 (2024).


import numpy as np
import scipy as sp
import heapq


def encode_ppr(r, r_bd, private_gen, alpha=2.0):
    """Perform encoding
    r: Function that gives the ratio dP/dQ
    r_bd: An upper-bound on the values of r
    private_gen: Private randomness generator
    alpha: parameter of the algorithm
    Returns: k where k is the index of the generated data
    """

    u = 0
    ws = np.inf
    k = 0
    ks = 0
    n = 0
    g1 = sp.special.gammainc(1 - 1 / alpha, 1) * sp.special.gamma(1 - 1 / alpha)
    h = []

    sprob = (1 / np.e) / (1 / np.e + g1)

    while True:
        u += private_gen.exponential()
        b = (u * alpha / (1 / np.e + g1)) ** alpha
        bpia = b ** (1 / alpha)

        if n == 0 and b * r_bd**-alpha >= ws:
            return ks

        if private_gen.random() < sprob:
            t = bpia
            v = private_gen.exponential() + 1
        else:
            v = 2
            while v > 1:
                v = private_gen.gamma(1 - 1 / alpha)

            t = bpia / v ** (1 / alpha)

        th = 1 if (t / r_bd) ** alpha * v <= ws else 0
        heapq.heappush(h, (t, v, th))
        n += th

        while h and h[0][0] <= bpia:
            t, v, th = heapq.heappop(h)
            n -= th
            k += 1
            ratio = r(k)
            w = (t / ratio) ** alpha * v
            if w < ws:
                (ws, ks) = (w, k)
