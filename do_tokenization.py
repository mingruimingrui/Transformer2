#!/usr/bin/env python

import os
import shutil
import logging
import argparse
import sentencepiece

from transformer_2_cli.train_config import make_config
from transformer_2_cli import setup_utils
from transformer_2.data.processing import make_processor_from_list
from transformer_2.utils.file_utils import map_file
from transformer_2.utils.io_utils \
    import read_yaml_from_file, write_yaml_to_file

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_file', metavar='CONFIG_FILE',
        help='A yaml or json file containing trainig configs')

    return parser.parse_args()


def do_tokenization(config):
    logger.info('Running {}'.format(os.path.basename(__file__)))

    # Ensure that clean corpus files are already created
    src_clean_corpus_path = setup_utils.get_data_filepath(
        config, 'clean', 'src')
    tgt_clean_corpus_path = setup_utils.get_data_filepath(
        config, 'clean', 'tgt')
    assert os.path.isfile(src_clean_corpus_path) \
        and os.path.isfile(tgt_clean_corpus_path), \
        'Make sure to run do_preprocessing first'

    output_dir = setup_utils.get_output_dir(config)
    for t in ['src', 'tgt']:
        clean_corpus_path = setup_utils.get_data_filepath(config, 'clean', t)
        token_corpus_path = setup_utils.get_data_filepath(config, 'token', t)
        spm_model_prefix = os.path.join(
            output_dir, 'spm_model.{}'.format(t))
        spm_model_path = '{}.model'.format(spm_model_prefix)
        spm_configs = config.src_spm_configs \
            if t == 'src' else config.tgt_spm_configs

        # Check if sentencepiece model needs to be trained
        if spm_configs.use_existing:
            logger.info('Copying spm_model from {}'.format(
                spm_configs.use_existing))
            shutil.copy(spm_configs.use_existing, spm_model_path)
        else:
            logger.info('Starting to train spm model')

            user_defined_symbols = spm_configs.user_defined_symbols
            if spm_configs.add_digit_tokens:
                # user_defined_symbols = ['▁{}'.format(i) for i in range(10)]
                user_defined_symbols += ['{}'.format(i) for i in range(10)]

            command = '--input={}'.format(clean_corpus_path)
            command += ' --model_prefix={}'.format(spm_model_prefix)
            command += ' --vocab_size={}'.format(spm_configs.vocab_size)
            command += ' --character_coverage={}'.format(
                spm_configs.character_coverage)
            command += ' --model_type={}'.format(spm_configs.model_type)
            command += ' --input_sentence_size={}'.format(
                spm_configs.input_sentence_size)
            command += ' --bos_id=0'
            command += ' --pad_id=1'
            command += ' --eos_id=2'
            command += ' --unk_id=3'  # This is the same order as fairseq
            if len(user_defined_symbols) > 0:
                command += ' --user_defined_symbols={}'.format(
                    ','.join(user_defined_symbols))
            sentencepiece.SentencePieceTrainer.Train(command)

        # Attempt to load spm model
        logger.info('Attempting to load {} spm model'.format(t))
        spm_model = sentencepiece.SentencePieceProcessor()
        spm_model.load(spm_model_path)

        # Check that sentencepiece model has non negative
        # bos, pad, eos, and unk tokens
        assert spm_model.bos_id() >= 0 \
            and spm_model.pad_id() >= 0 \
            and spm_model.eos_id() >= 0 \
            and spm_model.unk_id() >= 0, (
            'sentencepiece model at {} needs to have positive '
            'bos, pad, eos, and unk tokens, please retrain model'
        ).format(spm_model_path)

        # Do spm tokenization
        logger.info('Starting {} spm tokenization'.format(t))
        processing_steps = [{'spm_encode': {
            'spm_model_path': spm_model_path,
            'form': 'pieces'
        }}]
        processor = make_processor_from_list(processing_steps)
        map_file(
            fn=processor,
            in_filepath=clean_corpus_path,
            out_filepath=token_corpus_path,
            mode='w',
            num_workers=config.num_workers,
            show_pbar=True)

    # Save preprocessing steps to cached config file
    cached_config_path = setup_utils.get_cached_config_path(config)
    cached_config = read_yaml_from_file(cached_config_path)
    cached_config['src_spm_configs'] = config.src_spm_configs
    cached_config['tgt_spm_configs'] = config.tgt_spm_configs
    write_yaml_to_file(cached_config, cached_config_path)


def main():
    args = parse_args()
    config = make_config(config_file=args.config_file)
    setup_utils.do_setup(
        config,
        logger_or_name=logger,
        logfile_prefix='tokenization'
    )
    do_tokenization(config)


if __name__ == "__main__":
    main()
