#!/usr/bin/env python3
"""Synthetic gallery with realistic geometry. Build plan 5.1.

Uniform random unit vectors in 512 dimensions are all nearly orthogonal, which
makes ANY index look artificially good: nearest-neighbour search is trivial
when nothing is close to anything. Real ReID embeddings are CLUSTERED -- each
identity is a centroid plus intra-class jitter -- so that is what is generated
here. An index benchmarked on uniform noise is a benchmark of nothing.
"""
import pathlib, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
n = int(sys.argv[1]); dim = 512
rng = np.random.default_rng(42)
n_clusters = max(8, n // 200)
centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
assign = rng.integers(0, n_clusters, n)
V = centers[assign] + rng.normal(scale=0.35, size=(n, dim)).astype(np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True)
d = REPO / "data" / "bench"; d.mkdir(parents=True, exist_ok=True)
np.save(d / f"gallery_{n}.npy", V.astype(np.float32))
print(f"  {n} vectors, {n_clusters} clusters, {V.nbytes/1e6:.1f} MB fp32, "
      f"{V.nbytes/2e6:.1f} MB fp16")
