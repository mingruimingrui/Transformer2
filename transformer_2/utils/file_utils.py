""" Some helper functions to work with text files """

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import os
import math
import subprocess
import multiprocessing

from tqdm import tqdm
from six import integer_types

from transformers_2.utils.io_utils import open_txt_file

__all__ = ['count_lines', 'map_file']


def count_lines(filepath):
    command = 'cat {} | wc -l'.format(filepath)
    num_lines = int(subprocess.check_output(command, shell=True))
    return num_lines


def _map_file_worker_fn(
    fn, in_filepath, out_filepath,
    start_line, end_line,
    show_pbar=False
):
    with open_txt_file(out_filepath, 'w') as fout:
        with open_txt_file(in_filepath, 'r') as fin:
            for _ in range(start_line):
                fin.readline()
            iterable = range(end_line - start_line)
            if show_pbar:
                desc = 'Processing {}'.format(in_filepath)
                iterable = tqdm(iterable, desc=desc)
            for _ in iterable:
                line = fin.readline()
                fout.write(fn(line.strip()))
                fout.write('\n')
    return


def map_file(
    fn, in_filepath, out_filepath,
    mode='w', num_workers=0, show_pbar=True,
):
    """
    Apply a function on every line of an input file and prints output to an
    output file. Allows for the use of multi-processing and keeps memory
    usage to a minimum.

    Args:
        fn: A function that takes a string and outputs a string.
            This function will be applied to every line in in_filepath
        in_filepath: The path to the input file
        out_filepath: The path to the output file
        mode: The mode that output file should be opened, either one of [w, a]
        num_workers: The number of workers to use
        show_pbar: Should a progress bar be used?
            This progress bar will only track the progress of the first
            worker spawned.
    """
    assert mode in ['w', 'a']

    num_lines = count_lines(in_filepath)
    assert isinstance(num_workers, integer_types) and num_workers >= 0, \
        'num_workers should be a non-negative integer'

    print('Beginning the processing of {}'.format(in_filepath))
    print('This can take a while...')
    if num_workers == 0:
        _map_file_worker_fn(
            fn, in_filepath, out_filepath,
            start_line=0, end_line=num_lines,
            show_pbar=show_pbar
        )
        print('Done processing {}, view output in {}'.format(
            in_filepath, out_filepath))
        return

    # Initialize each worker to process only a chunk of the entire corpus
    chunk_size = int(math.ceil(float(num_lines) / num_workers))
    processes = []
    temp_out_filepaths = []
    for i in range(num_workers):
        temp_out_filepath = '{}.{}'.format(out_filepath, i)
        temp_out_filepaths.append(temp_out_filepath)

        # Determine start and end line for this worker
        start_line = chunk_size * i
        start_line = min(start_line, num_lines)
        end_line = start_line + chunk_size
        end_line = min(end_line, num_lines)

        p = multiprocessing.Process(
            target=_map_file_worker_fn,
            kwargs={
                'fn': fn,
                'in_filepath': in_filepath, 'out_filepath': temp_out_filepath,
                'start_line': start_line, 'end_line': end_line,
                'show_pbar': show_pbar and (i == 0)
            }
        )
        p.start()
        processes.append(p)

    # Wait for each worker to finish their work
    for p in processes:
        p.join()

    # Join all output files
    print('Gathering all temp output files')
    if mode == 'w' and os.path.isfile(out_filepath):
        os.remove(out_filepath)  # Remove old file if write mode
    for temp_out_filepath in temp_out_filepaths:
        command = 'cat {} >> {}'.format(temp_out_filepath, out_filepath)
        subprocess.call(command, shell=True)
        subprocess.call('rm {}'.format(temp_out_filepath), shell=True)

    print('Done processing {}, view output in {}'.format(
        in_filepath, out_filepath))
