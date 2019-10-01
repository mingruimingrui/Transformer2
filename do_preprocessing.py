#!/usr/bin/env python

import os
import logging
import argparse
from six import string_types

from transformer_2.data.processing import make_processor_from_list
from transformer_2.utils.file_utils import count_lines, map_file
from transformer_2.utils.io_utils \
    import read_yaml_from_file, write_yaml_to_file

from transformer_2_cli.train_config import make_config
from transformer_2_cli import setup_utils

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_file', metavar='CONFIG_FILE',
        help='A yaml or json file containing trainig configs')

    return parser.parse_args()


def do_preprocessing(config):
    logger.info('Running {}'.format(os.path.basename(__file__)))

    # Ensure that src_corpus_paths and tgt_corpus_paths are in list form
    src_corpus_paths = config.src_corpus_paths
    tgt_corpus_paths = config.tgt_corpus_paths
    if isinstance(src_corpus_paths, string_types):
        src_corpus_paths = [src_corpus_paths]
    if isinstance(tgt_corpus_paths, string_types):
        tgt_corpus_paths = [tgt_corpus_paths]
    assert isinstance(src_corpus_paths, list)
    assert isinstance(tgt_corpus_paths, list)

    # Make processor
    logger.info('Attempting to make source and target language processors')
    src_lang_processor = make_processor_from_list(
        config.src_preprocessing_steps)
    tgt_lang_processor = make_processor_from_list(
        config.tgt_preprocessing_steps)

    for i, (src_corpus_path, tgt_corpus_path) in enumerate(zip(
        src_corpus_paths, tgt_corpus_paths
    )):
        logger.info('Processing {} and {}'.format(
            src_corpus_path, tgt_corpus_path))
        assert isinstance(src_corpus_path, string_types), \
            'Expecting a path, got {}'.format(src_corpus_path)
        assert isinstance(tgt_corpus_path, string_types), \
            'Expecting a path, got {}'.format(tgt_corpus_path)
        assert os.path.isfile(src_corpus_path), \
            'Unable to find file {}'.format(src_corpus_path)
        assert os.path.isfile(tgt_corpus_path), \
            'Unable to find file {}'.format(tgt_corpus_path)

        # Ensuring that both corpus has same number of lines
        logger.info('Ensuring that both corpus has equal number of lines')
        assert count_lines(src_corpus_path) == count_lines(tgt_corpus_path), (
            '{} and {} has different number of lines'
        ).format(src_corpus_path, tgt_corpus_path)

        # Start cleaning corpus with multiple workers
        src_clean_corpus_path = setup_utils.get_data_filepath(
            config, 'clean', 'src')
        tgt_clean_corpus_path = setup_utils.get_data_filepath(
            config, 'clean', 'tgt')

        map_file(
            src_lang_processor,
            in_filepath=src_corpus_path,
            out_filepath=src_clean_corpus_path,
            mode='w' if i == 0 else 'a',
            num_workers=config.num_workers,
            show_pbar=True)
        map_file(
            tgt_lang_processor,
            in_filepath=tgt_corpus_path,
            out_filepath=tgt_clean_corpus_path,
            mode='w' if i == 0 else 'a',
            num_workers=config.num_workers,
            show_pbar=True)

    # Save preprocessing steps to cached config file
    cached_config_path = setup_utils.get_cached_config_path(config)
    cached_config = read_yaml_from_file(cached_config_path)
    cached_config['src_preprocessing_steps'] = config.src_preprocessing_steps
    cached_config['tgt_preprocessing_steps'] = config.tgt_preprocessing_steps
    write_yaml_to_file(cached_config, cached_config_path)


def main():
    args = parse_args()
    config = make_config(config_file=args.config_file)
    setup_utils.do_setup(
        config,
        logger_or_name=logger,
        logfile_prefix='preprocess'
    )
    do_preprocessing(config)


if __name__ == "__main__":
    main()
