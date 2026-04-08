# Sparse Randomized Response

Code accompanying the paper:

**Communication-Efficient Publication of Sparse Vectors under Differential Privacy via Poisson Private Representation**  
Quentin Hillebrand, Vorapong Suppakitpaisarn, and Tetsuo Shibuya  
ASIA CCS 2026

This repository implements a communication-efficient way to publish **sparse vectors** under **metric differential privacy** by compressing randomized response with a Poisson Private Representation (PPR)-based method. The main goal is to make private publication practical when vectors are very high-dimensional but contain only a small number of non-trivial entries.

The method is designed for settings where the natural non-private representation is already sparse, such as:

- social-network adjacency vectors,
- user-item interaction vectors in recommender systems,
- SNP / genomic variation profiles.

The proposed method reduces communication from dependence on the ambient dimension to dependence on the number of non-trivial entries, while preserving the exact randomized-response output distribution up to the privacy conversion introduced by PPR.

---

## Repository Overview

The `src/` directory currently contains the core implementation and experiment scripts:

- `compressed_randomized_response.py` — sparse-vector compression and decoding utilities
- `poisson_private_representation.py` — modified PPR encoder used by the project
- `counter_based_prng.py` — counter-based randomness utilities
- `compressed_graph.py` — compressed graph / adjacency-list machinery
- `graph.py` — graph loading and preprocessing helpers
- `recommender.py` — recommendation-system experiments
- `dna.py` — SNP / genomic-data experiments
- `triangles.py` — triangle-counting experiments on compressed graph data
- `distance.py` — distance / evaluation helpers

The repository also includes a `requirements.txt` with Python dependencies for numerical computing, graphs, plotting, and symbolic utilities.

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{hillebrand2026sparserr,
  author    = {Quentin Hillebrand and Vorapong Suppakitpaisarn and Tetsuo Shibuya},
  title     = {Communication-Efficient Publication of Sparse Vectors under Differential Privacy via Poisson Private Representation},
  booktitle = {Proceedings of the ACM Asia Conference on Computer and Communications Security (ASIA CCS '26)},
  year      = {2026},
  doi       = {10.1145/3779208.3805989}
}
```

---

## Acknowledgment

This repository builds on Poisson Private Representation (PPR).
The implementation in src/poisson_private_representation.py is adapted from:

Liu, Yanxiao, Wei-Ning Chen, Ayfer Özgür, and Cheuk Ting Li.
**Universal Exact Compression of Differentially Private Mechanisms.** 2024.
