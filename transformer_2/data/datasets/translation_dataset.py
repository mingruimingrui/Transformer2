"""
Likely the most over-engineered piece of code in this entire repository
"""

from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import shutil
import atexit
import warnings

import base64
import numpy as np

import sentencepiece

import tensorflow as tf

import threading
import multiprocessing

from transformer_2.utils.file_utils import count_lines, map_file
from transformer_2.utils.io_utils \
    import read_json_from_file, write_json_to_file
from transformer_2.utils.coordinator import CoordinatorStoppedException, \
    Coordinator, coordinated_get, coordinated_put
from transformer_2.utils.batching import batch_tokenized_pairs

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

BASE64_BUFFER_DTYPE = np.uint16


class SpmBinarizer(object):
    """
    Dummy class that would create a callable that can store internal states
    """

    def __init__(self, spm_model_path, corpus_format='raw'):
        assert corpus_format in ['raw', 'piece', 'id'], (
            'corpus_format must be one of [raw, piece, id], got {}'
        ).format(corpus_format)

        self.spm_model = sentencepiece.SentencePieceProcessor()
        self.spm_model.Load(spm_model_path)
        self.corpus_format = corpus_format

    def __call__(self, text):
        if self.corpus_format == 'raw':
            output = self.spm_model.EncodeAsIds(text)
        elif self.corpus_format == 'piece':
            pieces = text.split()
            output = [self.spm_model.PieceToId(p) for p in pieces]
        else:
            # corpus_format == 'id'
            output = [int(i) for i in text.split()]
        output = base64.b64encode(np.array(output, dtype=BASE64_BUFFER_DTYPE))
        return output.decode('utf-8')  # output needs to be unicode


def _reader_batcher_subprocess_loop(
    coord,
    shared_dict,
    chunk_queue,
    corpus_num_lines,
    src_corpus_path,
    tgt_corpus_path,
    src_spm_model,
    tgt_spm_model,
    max_batch_tokens=1000,
    max_batch_sentences=100,
    shuffle=False,
    buffer_size=100000,
    do_optimal_batching=False,
    generate_infinitely=False,
    warn_on_skip=True
):
    """ """
    # If buffer_size == 0, then use entire corpus as 1 chunk
    if buffer_size == 0:
        buffer_size = corpus_num_lines
    buffer_size = min(corpus_num_lines, buffer_size)

    # Open files and create the get_next_chunk helper function to get the next
    # chunk of dataset samples
    corpus_files = [
        open(src_corpus_path, 'rb'),
        open(tgt_corpus_path, 'rb')
    ]

    def close_corpus_files():
        corpus_files[0].close()
        corpus_files[1].close()

    def reopen_corpus_files():
        close_corpus_files()
        corpus_files[0] = open(src_corpus_path, 'rb')
        corpus_files[1] = open(tgt_corpus_path, 'rb')

    def get_next_chunk():
        done = False
        buffer = []
        for _ in range(buffer_size):
            src_line = corpus_files[0].readline().strip()
            tgt_line = corpus_files[1].readline().strip()

            # Determine if the end of corpus has been reached
            if len(src_line) == 0 or len(tgt_line) == 0:
                if not generate_infinitely:
                    done = True
                    break
                reopen_corpus_files()
                src_line = corpus_files[0].readline().strip()
                tgt_line = corpus_files[1].readline().strip()

            # Decode token ids from line
            # then append bos and eos tokens
            src_line = base64.b64decode(src_line)
            src_token_ids = np.frombuffer(src_line, dtype=BASE64_BUFFER_DTYPE)
            src_token_ids = np.concatenate([
                [src_spm_model.bos_id()],
                src_token_ids,
                [src_spm_model.eos_id()]
            ])

            tgt_line = base64.b64decode(tgt_line)
            tgt_token_ids = np.frombuffer(tgt_line, dtype=BASE64_BUFFER_DTYPE)
            tgt_token_ids = np.concatenate([
                [tgt_spm_model.bos_id()],
                tgt_token_ids,
                [tgt_spm_model.eos_id()]
            ])

            # Skip if too long
            if (
                len(src_token_ids) > max_batch_tokens or
                len(tgt_token_ids) > max_batch_tokens or
                len(src_token_ids) > 1024 or
                len(tgt_token_ids) > 1024
            ):
                if warn_on_skip:
                    warnings.warn('Skipping sample as it is too long')
                continue
            buffer.append((src_token_ids, tgt_token_ids))
        return buffer, done

    num_batches_generated = 0
    done = False
    while not done and not coord.should_stop():
        # Get next chunk of sentences
        buffer, done = get_next_chunk()

        # Form chunk into batches
        _, batches, _, _ = batch_tokenized_pairs(
            list_of_tokenized_pairs=buffer,
            src_padding_idx=src_spm_model.pad_id(),
            tgt_padding_idx=tgt_spm_model.pad_id(),
            max_batch_tokens=max_batch_tokens,
            max_batch_sentences=max_batch_sentences,
            shuffle=shuffle,
            do_optimal_batching=do_optimal_batching
        )

        # Attempt to place into chunk queue
        try:
            coordinated_put(coord, chunk_queue, batches)
        except CoordinatorStoppedException:
            break
        num_batches_generated += len(batches)

    # Close corpus when done
    close_corpus_files()
    shared_dict['num_batches_generated'] = num_batches_generated
    if not coord.should_stop():
        coord.request_stop()


def _unwrapper_subthread_loop(coord, chunk_queue, batch_queue):
    while not coord.should_stop():
        try:
            batches = coordinated_get(coord, chunk_queue)
            for batch in batches:
                coordinated_put(coord, batch_queue, batch)
        except CoordinatorStoppedException:
            break


class TranslationDataset(object):

    def __init__(self, cache_dir: str):
        metadata_filepath = os.path.join(cache_dir, 'metadata.json')
        cache_metadata = read_json_from_file(metadata_filepath)

        self.corpus_num_lines = cache_metadata['num_lines']

        self.src_corpus_path = cache_metadata['src_corpus_path']
        self.tgt_corpus_path = cache_metadata['tgt_corpus_path']

        self.src_spm_model = sentencepiece.SentencePieceProcessor()
        self.src_spm_model.Load(cache_metadata['src_spm_model_path'])
        self.tgt_spm_model = sentencepiece.SentencePieceProcessor()
        self.tgt_spm_model.Load(cache_metadata['tgt_spm_model_path'])

        # Ensure that the number of tokens are within a reasonable limit
        max_token_limit = np.iinfo(BASE64_BUFFER_DTYPE).max + 1
        assert self.src_spm_model.get_piece_size() <= max_token_limit, (
            'Source sentencepiece model has too many tokens, '
            'dataset supports a maximum of {} tokens'
        ).format(max_token_limit)
        assert self.tgt_spm_model.get_piece_size() <= max_token_limit, (
            'Target sentencepiece model has too many tokens, '
            'dataset supports a maximum of {} tokens'
        ).format(max_token_limit)

    @classmethod
    def new(
        cls,
        cache_dir,
        src_corpus_path, tgt_corpus_path,
        src_spm_model_path, tgt_spm_model_path,
        corpus_format='raw', num_workers=0
    ):
        """
        Takes in a parallel language corpus for machine translation, process
        the corpus and cache to file

        Args:
            src_corpus_path: Path to the source language corpus, should be a
                text file.
            tgt_corpus_path: Path to the target language corpus, should be a
                text file.
            src_spm_model_path: Path to the source sentencepiece model
            tgt_spm_model_path: Path to the target sentencepiece model
            corpus_format: The format that source and target corpus are in
                - raw: Corpus is raw text format
                - piece: Corpus is already tokenized as pieces
                - id: Corpus is already tokenized as ids
        """
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        # Check spm model are valid
        def check_spm_model(spm_model_path):
            # Ensure that sentencepiece model can be loaded
            assert os.path.isfile(spm_model_path), \
                'Unable to find file {}'.format(spm_model_path)
            spm_model = sentencepiece.SentencePieceProcessor()
            spm_model.Load(spm_model_path)

            # Ensure that unk, eos, bos, and pad token are all valid
            def check_required_token(token_id, token_name):
                assert token_id >= 0, (
                    'Require {name} token to have a default token id of >= 0. '
                    'Got instead {name} token with token id of {token_id} '
                    'in {model_path}. Please retrain sentencepiece model '
                    'ensuring that the {name} token has an id of >= 0.'
                ).format(
                    name=token_name,
                    token_id=token_id,
                    model_path=spm_model_path
                )

            check_required_token(spm_model.unk_id(), 'unkown')
            check_required_token(spm_model.bos_id(), 'beginning of sentence')
            check_required_token(spm_model.eos_id(), 'ending of sentence')
            check_required_token(spm_model.pad_id(), 'padding')

        print('Ensuring that sentencepiece models are valid')
        check_spm_model(src_spm_model_path)
        check_spm_model(tgt_spm_model_path)
        src_spm_binarizer = SpmBinarizer(
            src_spm_model_path, corpus_format=corpus_format)
        tgt_spm_binarizer = SpmBinarizer(
            tgt_spm_model_path, corpus_format=corpus_format)

        # Check corpus files are valid
        assert os.path.isfile(src_corpus_path), \
            'Unable to find {}'.format(src_corpus_path)
        assert os.path.isfile(tgt_corpus_path), \
            'Unable to find {}'.format(tgt_corpus_path)
        print('Counting number of lines in src corpus')
        src_corpus_num_lines = count_lines(src_corpus_path)
        print('Counting number of lines in tgt corpus')
        tgt_corpus_num_lines = count_lines(tgt_corpus_path)
        assert src_corpus_num_lines == tgt_corpus_num_lines, (
            'Source and target corpus has different number of lines'
            'Please check the contents of {} and {}'
        ).format(src_corpus_path, tgt_corpus_path)

        # Cache some corpus metadata
        cache_metadata = {
            'num_lines': src_corpus_num_lines,
            'src_corpus_path': os.path.join(cache_dir, 'src_corpus.bin'),
            'tgt_corpus_path': os.path.join(cache_dir, 'tgt_corpus.bin'),
            'src_spm_model_path':
                os.path.join(cache_dir, 'src_spm_model.model'),
            'tgt_spm_model_path':
                os.path.join(cache_dir, 'tgt_spm_model.model'),
        }
        metadata_filepath = os.path.join(cache_dir, 'metadata.json')
        write_json_to_file(cache_metadata, metadata_filepath)

        # Copy spm models over
        shutil.copy(src_spm_model_path, cache_metadata['src_spm_model_path'])
        shutil.copy(tgt_spm_model_path, cache_metadata['tgt_spm_model_path'])

        # Binarize corpus files
        map_file(
            src_spm_binarizer,
            src_corpus_path, cache_metadata['src_corpus_path'],
            num_workers=num_workers, show_pbar=True)
        map_file(
            tgt_spm_binarizer,
            tgt_corpus_path, cache_metadata['tgt_corpus_path'],
            num_workers=num_workers, show_pbar=True)

        return TranslationDataset(cache_dir)

    def make_batch_generator(
        self,
        max_batch_tokens=1024,
        max_batch_sentences=1024,
        shuffle=False,
        buffer_size=10000,
        do_optimal_batching=False,
        generate_infinitely=False,
        warn_on_skip=True
    ):
        """
        There are 3 components to the batch_generator,
            - A reader + batcher on a sub-process
            - unwrapper on a sub-thread
            - and an interface on the main-thread
        """
        max_batch_tokens = int(max_batch_tokens)
        max_batch_sentences = int(max_batch_sentences)
        buffer_size = int(buffer_size)

        assert buffer_size >= 0, (
            'buffer_size should be a non-negative integer. '
            'To store entire corpus in memory, use buffer_size = 0'
        )

        # Create stared objects
        manager = multiprocessing.Manager()
        batcher_coord = Coordinator(manager)
        unwrapper_coord = Coordinator(manager)
        shared_dict = manager.dict()
        chunk_queue = manager.Queue(maxsize=1)
        batch_queue = Queue(maxsize=100)

        self.batcher_coord = batcher_coord
        self.unwrapper_coord = unwrapper_coord

        # Start reader + batcher sub-process
        batcher_subprocess = multiprocessing.Process(
            target=_reader_batcher_subprocess_loop,
            kwargs={
                'coord': batcher_coord,
                'shared_dict': shared_dict,
                'chunk_queue': chunk_queue,
                'corpus_num_lines': self.corpus_num_lines,
                'src_corpus_path': self.src_corpus_path,
                'tgt_corpus_path': self.tgt_corpus_path,
                'src_spm_model': self.src_spm_model,
                'tgt_spm_model': self.tgt_spm_model,
                'max_batch_tokens': max_batch_tokens,
                'max_batch_sentences': max_batch_sentences,
                'shuffle': shuffle,
                'buffer_size': buffer_size,
                'do_optimal_batching': do_optimal_batching,
                'generate_infinitely': generate_infinitely,
                'warn_on_skip': warn_on_skip
            }
        )

        # Start unwrapper sub-thread
        unwrapper_subthread = threading.Thread(
            target=_unwrapper_subthread_loop,
            kwargs={
                'coord': unwrapper_coord,
                'chunk_queue': chunk_queue,
                'batch_queue': batch_queue
            }
        )

        def stop_workers():
            batcher_coord.request_stop()
            unwrapper_coord.request_stop()
            batcher_subprocess.join()
            unwrapper_subthread.join()
        atexit.register(stop_workers)

        batcher_subprocess.start()
        unwrapper_subthread.start()

        # Consume batches from batch_queue and yield
        num_batches_yielded = 0
        while not batcher_coord.should_stop():
            try:
                batch = coordinated_get(batcher_coord, batch_queue)
            except CoordinatorStoppedException:
                break
            num_batches_yielded += 1
            yield batch

        # Yield all remaining batches in queue
        num_remaining_batches = \
            shared_dict['num_batches_generated'] - num_batches_yielded
        for _ in range(num_remaining_batches):
            batch = coordinated_get(unwrapper_coord, batch_queue)
            yield batch

        # Stop all workers
        stop_workers()

    def make_tf_dataset(
        self,
        max_batch_tokens=1024,
        max_batch_sentences=1024,
        shuffle=False,
        buffer_size=10000,
        do_optimal_batching=False,
        generate_infinitely=False,
        warn_on_skip=True
    ):
        tokens_tensor_shape = tf.TensorShape([None, None])
        length_tensor_shape = tf.TensorShape([None])
        return tf.data.Dataset.from_generator(
            generator=self.make_batch_generator,
            output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
            output_shapes=(
                tokens_tensor_shape, tokens_tensor_shape,
                length_tensor_shape, length_tensor_shape
            ),
            args=(
                max_batch_tokens,
                max_batch_sentences,
                shuffle,
                buffer_size,
                do_optimal_batching,
                generate_infinitely,
                warn_on_skip
            )
        )
