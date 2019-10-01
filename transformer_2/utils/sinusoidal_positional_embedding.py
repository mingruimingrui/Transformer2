from __future__ import division

import math
import numpy as np
from typing import Optional


def build_sinusoidal_positional_embedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    dtype=np.float32
):
    """
    Build sinusoidal embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = np.exp(-emb * np.arange(half_dim, dtype=dtype))
    emb = np.arange(num_embeddings, dtype=dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    emb = np.reshape(emb, [num_embeddings, -1])
    if embedding_dim % 2 == 1:
        # zero pad
        emb = np.concatenate(
            [emb, np.zeros(shape=[num_embeddings, 1], dtype=dtype)],
            axis=1
        )
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb
