"""
Configurables for training
"""

import os
from copy import deepcopy
from six import string_types, integer_types

from transformer_2.utils.config_system import ConfigSystem
from transformer_2.models.transformer_config import _C as _model_configs

__all__ = ['make_config']


def validate_config_fn(config):
    def check_type(obj_name, obj, expected_type):
        """ Check if obj is of expected type """
        assert isinstance(obj, expected_type), (
            'Expecting {} to be {} type, instead got {}'
        ).format(obj_name, expected_type, type(obj))

    def check_file_exists(filepath):
        assert os.path.isfile(filepath), \
            'Unable to find {}'.format(os.path.abspath(filepath))

    # General configs
    assert config.train_dir is not None, \
        'train_dir must be provided'
    check_type('train_dir', config.train_dir, string_types)
    check_type('num_workers', config.num_workers, integer_types)
    assert len(config.src_corpus_paths) > 0, \
        'Atleast 1 training corpus has to be provided'
    assert len(config.src_corpus_paths) == len(config.tgt_corpus_paths), \
        'Mismatched number of src and tgt training corpus'
    for filepath in config.src_corpus_paths:
        check_file_exists(filepath)
    for filepath in config.tgt_corpus_paths:
        check_file_exists(filepath)

    if config.src_valid_path is not None:
        assert config.tgt_valid_path is not None, (
            'If src_valid_path is provided, '
            'then tgt_valid_path must also be provided'
        )
        check_file_exists(config.src_valid_path)
        check_file_exists(config.tgt_valid_path)

    # Preprocessing configs
    if isinstance(config.src_preprocessing_steps, string_types):
        src_lang = config.src_preprocessing_steps
        from transformer_2.data.processing import DEFAULT_PROCESSING_STEPS
        assert src_lang in DEFAULT_PROCESSING_STEPS, \
            '{} does not have default processing steps'.format(src_lang)
        config.src_preprocessing_steps = DEFAULT_PROCESSING_STEPS[src_lang]
        print('Using default processing steps for {}'.format(src_lang))
        print(config.src_preprocessing_steps)

    if isinstance(config.tgt_preprocessing_steps, string_types):
        tgt_lang = config.tgt_preprocessing_steps
        from transformer_2.data.processing import DEFAULT_PROCESSING_STEPS
        assert tgt_lang in DEFAULT_PROCESSING_STEPS, \
            '{} does not have default processing steps'.format(tgt_lang)
        config.tgt_preprocessing_steps = DEFAULT_PROCESSING_STEPS[tgt_lang]
        print('Using default processing steps for {}'.format(tgt_lang))
        print(config.tgt_preprocessing_steps)

    # Tokenization configs
    if config.src_spm_configs.use_existing is not None:
        check_file_exists(config.src_spm_configs.use_existing)
    if config.tgt_spm_configs.use_existing is not None:
        check_file_exists(config.tgt_spm_configs.use_existing)

    # Train configs
    check_type('num_steps', config.train_configs.num_steps, integer_types)
    check_type('update_freq', config.train_configs.update_freq, integer_types)
    check_type(
        'max_batch_tokens',
        config.train_configs.max_batch_tokens,
        integer_types
    )
    check_type(
        'max_batch_sentences',
        config.train_configs.max_batch_sentences,
        integer_types
    )
    check_type('lr', config.train_configs.lr, float)
    check_type(
        'warmup_steps',
        config.train_configs.warmup_steps,
        integer_types
    )
    check_type('warmup_init_lr', config.train_configs.warmup_init_lr, float)
    assert config.train_configs.lr_scheduler == 'fixed', \
        'Currently only fixed learning rate is implemented'
    check_type('min_lr', config.train_configs.min_lr, float)
    assert len(config.train_configs.adam_betas) == 2
    check_type('adam_beta_1', config.train_configs.adam_betas[0], float)
    check_type('adam_beta_2', config.train_configs.adam_betas[1], float)
    if config.train_configs.clipnorm is not None:
        check_type('clipnorm', config.train_configs.clipnorm, float)

    # Log configs
    check_type(
        'log_interval',
        config.train_configs.log_interval,
        integer_types
    )
    check_type(
        'checkpoint_interval',
        config.train_configs.checkpoint_interval,
        integer_types
    )


_C = ConfigSystem(validate_config_fn=validate_config_fn)
# --------------------------------------------------------------------------- #
# Start of general configs
# --------------------------------------------------------------------------- #

# `train_dir` (type: int) (required)
# The directory to store training outputs and artifacts
_C.train_dir = None

# `num_workers` (type: int) (default: 2)
# How many workers should be used to perform processing of corpus
_C.num_workers = 2

# `src_corpus_paths` (type: list[str]) (required)
# List of paths to the source language corpus
_C.src_corpus_paths = []

# `tgt_corpus_paths` (type: list[str]) (required)
# List of paths to the target language corpus
_C.tgt_corpus_paths = []

# `src_valid_path` (type: list[str]) (default: None)
# Path to a source langauge validation file
_C.src_valid_path = None

# `tgt_valid_path` (type: list[str]) (default: None)
# Path to a target langauge validation file
_C.tgt_valid_path = None

# --------------------------------------------------------------------------- #
# Start of preprocessing configs
# --------------------------------------------------------------------------- #

# `src_preprocessing_steps` (type: string | list[processor_config])
# (default: 'en')
# The steps applied to the source language corpus to do data cleaning
# Either a list of processor configs
# or one of ['en', 'de', 'fr', 'zh']
_C.src_preprocessing_steps = 'en'

# `tgt_preprocessing_steps` (type: string | list[processor_config])
# (default: 'en)
# The steps applied to the target language corpus to do data cleaning
# Either a list of processor configs
# or one of ['en', 'de', 'fr', 'zh']
_C.tgt_preprocessing_steps = 'en'


# --------------------------------------------------------------------------- #
# Start of tokenization configs
# --------------------------------------------------------------------------- #

# `src_spm_configs` are a set of configs dealing
# with the way the source language is tokenized
_C.src_spm_configs = ConfigSystem()

# `use_existing` (type: string) (default: None)
# If an existing spm model should be used, please provide a path
# to the spm model file
_C.src_spm_configs.use_existing = None

# `vocab_size` (type: integer) (default: 8000)
# The number of unique tokens
# Recommended to set to a value of 8000, 16000 or 32000
_C.src_spm_configs.vocab_size = 8000

# `character_coverage` (type: float) (default: 0.9995)
# The minimum percentage of characters to cover without the need for
# <unk> token (aka unkown token)
_C.src_spm_configs.character_coverage = 0.9995

# `model_type` (type: string) (default: 'bpe')
# The type of sentencepiece training to perform
# Should be one of [bpe, unigram, word, char] default bpe
_C.src_spm_configs.model_type = 'bpe'

# `input_sentence_size` (type: integer) (default: 3000000)
# To ensure that sentencepiece can train in a resonable amount of time,
# a subset of sentences can be selected
# By default a maximum of 1e7 (10 million) sentences will be used, to use all
# sentences, set this value to 0
_C.src_spm_configs.input_sentence_size = 3000000

# `add_digit_tokens` (type: boolean) (default: False)
# Adds 0-9 to user_defined_symbols
_C.src_spm_configs.add_digit_tokens = False

# `user_defined_symbols` (type: list[str]) (default: [])
_C.src_spm_configs.user_defined_symbols = []

# The sentencepiece configs for the target language
# options are the same as for source language
_C.tgt_spm_configs = deepcopy(_C.src_spm_configs)


# --------------------------------------------------------------------------- #
# Model configs
# --------------------------------------------------------------------------- #

# This variable cannot be changed at this point of time
_C.model_type = 'Transformer'

# `model_configs` is the model config for the transformer model
_C.model_configs = deepcopy(_model_configs)


# --------------------------------------------------------------------------- #
# Training configs
# --------------------------------------------------------------------------- #

# `train_configs` are a set of configs dealing
# with training parameters like steps and batch size
_C.train_configs = ConfigSystem()

# `num_steps` (type: integer) (default: 100000)
# The number of steps to train model for.
_C.train_configs.num_steps = 100000

# `update_freq` (type: integer) (default: 1)
# Gradients will be accumulated and updated every this number of steps/
_C.train_configs.update_freq = 1

# `max_batch_tokens` (type: integer) (default: 1024)
# The maxinum number of tokens to use per batch.
_C.train_configs.max_batch_tokens = 1024

# `max_batch_sentences` (type: integer) (default: 1024)
# The maxinum number of sentences to use per batch.
_C.train_configs.max_batch_sentences = 1024

# # `train_configs` (type: string) (default: 'cross_entropy')
# # The type of loss function to use.
# # Option of [cross_entropy]
# _C.train_configs.criterion = 'cross_entropy'

# `lr` (type: float) (default: 0.00001)
# The learning rate.
_C.train_configs.lr = 0.00001

# `warmup_steps` (type: int) (default: 16000)
# The number of steps to do optimizer warmup.
_C.train_configs.warmup_steps = 16000

# `warmup_init_lr` (type: float) (default: 1e-7)
# The initial learning rate for the warmup period.
_C.train_configs.warmup_init_lr = 1e-7

# `lr_scheduler` (type: string) (default: 'fixed')
# (options: ['fixed', 'inverse_time', 'inverse_sqrt'])
# The way learning rate changes during training
# after warm up.
_C.train_configs.lr_scheduler = 'fixed'

# `min_lr` (type: float) (default: 1e-9)
# The minimum learning rate as a result lr scheduler.
_C.train_configs.min_lr = 1e-9

# `adam_betas` (type: Tuple[float, float]) (default: '[0.9, 0.98]')
# The adam beta1 and beta2.
_C.train_configs.adam_betas = [0.9, 0.98]

# `clipnorm` (type: float) (default: None)
# Should gradient be clipped by norm of each tensor?
# This should be the clipnorm value.
_C.train_configs.clipnorm = None

# `fp16` (type: boolean) (default: False)
# Should fp16 training be used?
_C.train_configs.fp16 = False

# `dropout` (type: float) (default: 0.0)
# Dropout probability
_C.train_configs.dropout = None

# `dropout` (type: float) (default: 0.0)
# Dropout probability for attention weights
_C.train_configs.attn_dropout = None

# `dropout` (type: float) (default: 0.0)
# Dropout probability after attention in transistor
_C.train_configs.activation_dropout = None


# --------------------------------------------------------------------------- #
# Logging configs
# --------------------------------------------------------------------------- #

# `log_configs` are a set of configs dealing
# with logging matters
_C.log_configs = ConfigSystem()

# `log_interval` (type: int) (default: 60)
# The interval in seconds to do logging
_C.log_configs.log_interval = 60

# `checkpoint_interval` (type: int) (default: 7200)
# The interval in seconds to do logging
_C.log_configs.checkpoint_interval = 7200


# --------------------------------------------------------------------------- #
# End of configs
# --------------------------------------------------------------------------- #
_C.immutable(True)
make_config = _C.make_config
