from __future__ import absolute_import, unicode_literals, print_function

import random
import numpy as np
from typing import List, Tuple


def pad_sample(sample, width, padding_token):
    """ Left pad a sample to a given width """
    sample = np.array(sample)
    padding = [padding_token] * (width - len(sample))
    return np.concatenate([padding, sample]).astype(sample.dtype)


def form_batch_into_array(batch, max_width, padding_token):
    batch = [pad_sample(s, max_width, padding_token) for s in batch]
    return np.array(batch)


# def batch_tokenized_sents(
#     tokenized_sents: List[List[int]],
#     padding_idx: int,
#     max_positions: int = 1024,
#     max_batch_tokens: int = 1000,
#     max_batch_sentences: int = 100,
#     shuffle: bool = False,
#     do_optimal_batching: bool = False
# ):
#     sent_lengths = [len(token_ids) for token_ids in tokenized_sents]

#     # If optimal batching has to be done, then we first sort the tokenized
#     # pairs by the length of the source sequence
#     if do_optimal_batching:
#         all_idxs = np.argsort(-np.array(sent_lengths))
#     else:
#         all_idxs = range(len(tokenized_sents))

#     # Create placeholders for all batches and cur batch
#     batches_idxs = []
#     ignored_idxs = []

#     cur_batch_idxs = []
#     cur_batch_max_width = 0

#     for i in all_idxs:
#         sample_width = len(tokenized_sents[i])

#         # Ignore sample if too long or is empty
#         if (
#             sample_width > max_batch_tokens or
#             sample_width > max_positions or
#             sample_width == 0
#         ):
#             ignored_idxs.append(i)
#             continue

#         # Determine if new sample can be appended to cur batch
#         expected_batch_size = (len(cur_batch_idxs) + 1) * \
#             max(cur_batch_max_width, sample_width)
#         should_not_append_new_sample_to_cur_batch = \
#             len(cur_batch_idxs) >= max_batch_sentences or \
#             expected_batch_size > max_batch_tokens
#         if should_not_append_new_sample_to_cur_batch:
#             # Append cur batch to batches
#             batches_idxs.append(cur_batch_idxs)
#             # Reinit cur batch
#             cur_batch_idxs = []
#             cur_batch_max_width = 0

#         # Append cur sample to cur batch
#         cur_batch_idxs.append(i)
#         cur_batch_max_width = max(cur_batch_max_width, sample_width)

#     # Append last batch if non empty
#     if len(cur_batch_idxs) > 0:
#         batches_idxs.append(cur_batch_idxs)

#     # Shuffle batches if needed
#     if shuffle:
#         random.shuffle(batches_idxs)

#     # Gather batches and batches_lengths from idxs
#     batches = []
#     batches_lengths = []

#     for idxs in batches_idxs:
#         cur_batch_lengths = [sent_lengths[i] for i in idxs]
#         cur_batch = form_batch_into_array(
#             batch=[tokenized_sents[i] for i in idxs],
#             max_width=max(cur_batch_lengths),
#             padding_token=padding_idx
#         )

#         batches.append(cur_batch)
#         batches_lengths.append(cur_batch_lengths)

#     ignored_tokenized_sents = [tokenized_sents[i] for i in ignored_idxs]

#     return (
#         batches_idxs, batches, batches_lengths,
#         ignored_idxs, ignored_tokenized_sents
#     )


# def unbatch_tokenized_sents(
#     batches_idxs, batches,
#     ignored_idxs, ignored_tokenized_sents
# ):
#     # Get total number of sentences
#     total_num_sents = len(ignored_idxs)
#     for idxs in batches_idxs:
#         total_num_sents += len(idxs)

#     # Make placeholder to hold unbatched samples
#     ordered_samples = [None] * total_num_sents

#     # Unsort batches
#     for idxs, batch in zip(batches_idxs, batches):
#         for i, obj in zip(idxs, batch):
#             ordered_samples[i] = obj

#     # Fill ignore
#     for i, obj in zip(ignored_idxs, ignored_tokenized_sents):
#         ordered_samples[i] = obj

#     return ordered_samples


def batch_tokenized_pairs(
    list_of_tokenized_pairs: List[Tuple[List[int], List[int]]],
    src_padding_idx: int,
    tgt_padding_idx: int,
    max_src_positions: int = 1024,
    max_tgt_positions: int = 1024,
    max_batch_tokens: int = 1000,
    max_batch_sentences: int = 100,
    shuffle: bool = False,
    do_optimal_batching: bool = False
):
    # We're going to do some indexing make sure this is a list,
    # not generator/iterator
    list_of_tokenized_pairs = list(list_of_tokenized_pairs)

    # Gather sentence lengths
    src_sent_lengths = []
    tgt_sent_lengths = []
    for src_token_ids, tgt_token_ids in list_of_tokenized_pairs:
        src_sent_lengths.append(len(src_token_ids))
        tgt_sent_lengths.append(len(tgt_token_ids))

    # If optimal batching has to be done, then we first sort the tokenized
    # pairs by the length of the source sequence
    if do_optimal_batching:
        all_idxs = np.argsort(-np.array(src_sent_lengths))
    else:
        all_idxs = range(len(list_of_tokenized_pairs))

    # Create placeholders for all batches and cur batch
    batches_idxs = []
    ignored_idxs = []

    cur_batch_idxs = []
    cur_batch_src_max_width = 0
    cur_batch_tgt_max_width = 0

    for i in all_idxs:
        src_width = src_sent_lengths[i]
        tgt_width = tgt_sent_lengths[i]

        # Ignore sample if too long or is empty
        if (
            src_width + tgt_width > max_batch_tokens or
            src_width > max_src_positions or src_width == 0 or
            tgt_width > max_tgt_positions or tgt_width == 0
        ):
            ignored_idxs.append(i)
            continue

        # Determine if new sample can be appended to current batch
        expected_src_max_width = max(cur_batch_src_max_width, src_width)
        expected_tgt_max_width = max(cur_batch_tgt_max_width, tgt_width)
        expected_batch_size = (len(cur_batch_idxs) + 1) * \
            (expected_src_max_width + expected_tgt_max_width)
        if (
            len(cur_batch_idxs) >= max_batch_sentences or
            expected_batch_size > max_batch_tokens
        ):
            # Append cur batch to batches
            batches_idxs.append(cur_batch_idxs)

            # Re-initialize current batch
            cur_batch_idxs = []
            cur_batch_src_max_width = 0
            cur_batch_tgt_max_width = 0

        # Append current sample to current batch
        cur_batch_idxs.append(i)
        cur_batch_src_max_width = max(cur_batch_src_max_width, src_width)
        cur_batch_tgt_max_width = max(cur_batch_tgt_max_width, tgt_width)

    # Append last batch if non empty
    if len(cur_batch_idxs) > 0:
        batches_idxs.append(cur_batch_idxs)

    # Shuffle batches if needed
    if shuffle:
        random.shuffle(cur_batch_idxs)

    # Gather batches from idxs
    batches = []
    for idxs in batches_idxs:
        src_batch_lengths = [src_sent_lengths[i] for i in idxs]
        tgt_batch_lengths = [tgt_sent_lengths[i] for i in idxs]

        src_batch_tokens = form_batch_into_array(
            batch=[list_of_tokenized_pairs[i][0] for i in idxs],
            max_width=max(src_batch_lengths), padding_token=src_padding_idx
        )
        tgt_batch_tokens = form_batch_into_array(
            batch=[list_of_tokenized_pairs[i][1] for i in idxs],
            max_width=max(tgt_batch_lengths), padding_token=tgt_padding_idx
        )

        batches.append((
            src_batch_tokens, tgt_batch_tokens,
            src_batch_lengths, tgt_batch_lengths
        ))

    ignored_tokenized_pairs = \
        [list_of_tokenized_pairs[i] for i in ignored_idxs]

    return (batches_idxs, batches, ignored_idxs, ignored_tokenized_pairs)
