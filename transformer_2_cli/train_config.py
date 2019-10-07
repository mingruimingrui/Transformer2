"""
Configurables for training
"""

from copy import deepcopy
from transformer_2.utils.config_system import ConfigSystem

__all__ = ['make_config']


def validate_config_fn(config):
    return


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

# `src_preprocessing_steps` (type: list[processor_config]) (default: None)
# The steps applied to the source language corpus to do data cleaning
_C.src_preprocessing_steps = []

# `tgt_preprocessing_steps` (type: list[processor_config]) (default: None)
# The steps applied to the target language corpus to do data cleaning
_C.tgt_preprocessing_steps = []


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
_C.model_configs = ConfigSystem()


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
