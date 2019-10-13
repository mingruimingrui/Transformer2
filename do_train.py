#!/usr/bin/env python

import os
import sys
import shutil
import logging
import argparse
import warnings
from time import time
from tqdm import tqdm

import tensorflow as tf

from transformer_2.models import Transformer
from transformer_2.data.datasets import TranslationDataset
from transformer_2.losses.label_smoothed_nll_loss \
    import label_smoothed_nll_loss
from transformer_2.utils.io_utils import write_yaml_to_file

from transformer_2_cli import setup_utils
from transformer_2_cli.train_config import make_config

logger = logging.getLogger()

# Set memory growth on GPU
for d in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(d, True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_file', metavar='CONFIG_FILE',
        help='A yaml or json file containing trainig configs')

    return parser.parse_args()


def load_datasets(config):
    """ Load a tf dataset iterator """
    cache_dir = setup_utils.get_data_bin_dir(config)
    src_corpus_path = setup_utils.get_data_filepath(config, 'token', 'src')
    tgt_corpus_path = setup_utils.get_data_filepath(config, 'token', 'tgt')
    src_spm_model_path = setup_utils.get_spm_model_path(config, 'src')
    tgt_spm_model_path = setup_utils.get_spm_model_path(config, 'tgt')

    if os.path.isdir(cache_dir):
        dataset = TranslationDataset(cache_dir)
    else:
        dataset = TranslationDataset.new(
            cache_dir=cache_dir,
            src_corpus_path=src_corpus_path,
            tgt_corpus_path=tgt_corpus_path,
            src_spm_model_path=src_spm_model_path,
            tgt_spm_model_path=tgt_spm_model_path,
            corpus_format='piece',
            num_workers=config.num_workers
        )

    valid_dataset = None
    if config.src_valid_path is not None:
        valid_cache_dir = setup_utils.get_data_valid_bin_dir(config)
        src_corpus_path = setup_utils.get_data_filepath(config, 'valid', 'src')
        tgt_corpus_path = setup_utils.get_data_filepath(config, 'valid', 'tgt')

        if os.path.isdir(valid_cache_dir):
            valid_dataset = TranslationDataset(valid_cache_dir)
        else:
            valid_dataset = TranslationDataset.new(
                cache_dir=valid_cache_dir,
                src_corpus_path=src_corpus_path,
                tgt_corpus_path=tgt_corpus_path,
                src_spm_model_path=src_spm_model_path,
                tgt_spm_model_path=tgt_spm_model_path,
                corpus_format='piece',
                num_workers=config.num_workers
            )

    return dataset, valid_dataset


def make_tf_dataset_iterator(dataset, config, generate_infinitely=True):
    tf_dataset = dataset.make_tf_dataset(
        max_batch_tokens=config.train_configs.max_batch_tokens,
        max_batch_sentences=config.train_configs.max_batch_sentences,
        shuffle=True,
        buffer_size=1000000,
        do_optimal_batching=True,
        generate_infinitely=generate_infinitely,
        warn_on_skip=False
    ).prefetch(10)
    return iter(tf_dataset)


def make_dataset_generator(dataset, config, generate_infinitely=False):
    return dataset.make_batch_generator(
        max_batch_tokens=config.train_configs.max_batch_tokens,
        max_batch_sentences=config.train_configs.max_batch_sentences,
        shuffle=True,
        buffer_size=1000000,
        do_optimal_batching=True,
        generate_infinitely=generate_infinitely,
        warn_on_skip=False
    )


def load_model_and_optimizer(dataset, train_dtype, config):
    model_config = config.model_configs.to_dict()
    model_config['encoder_vocab_size'] = config.src_spm_configs.vocab_size
    model_config['encoder_padding_idx'] = dataset.src_spm_model.pad_id()
    model_config['decoder_vocab_size'] = config.tgt_spm_configs.vocab_size
    model_config['decoder_padding_idx'] = dataset.tgt_spm_model.pad_id()
    model_config['dropout'] = config.train_configs.dropout
    model_config['attn_dropout'] = config.train_configs.attn_dropout
    model_config['activation_dropout'] = \
        config.train_configs.activation_dropout
    model = Transformer.make_model(config_dict=model_config, dtype=train_dtype)

    optimizer_kwargs = {}
    if config.train_configs.clipnorm is not None:
        optimizer_kwargs['clipnorm'] = config.train_configs.clipnorm
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.train_configs.warmup_init_lr,
        beta_1=config.train_configs.adam_betas[0],
        beta_2=config.train_configs.adam_betas[1],
        **optimizer_kwargs
    )

    return model, optimizer


def compute_learning_rate(update_nb, config):
    lr = config.train_configs.lr
    lr_scheduler = config.train_configs.lr_scheduler
    warmup_steps = config.train_configs.warmup_steps
    warmup_init_lr = config.train_configs.warmup_init_lr

    if update_nb < warmup_steps:
        return (lr - warmup_init_lr) / warmup_steps * update_nb

    elif lr_scheduler == 'fixed':
        return config.train_configs.lr

    else:
        raise ValueError(
            'lr_scheduler {} not implemented yet'.format(lr_scheduler))


def save_model_config(model, config):
    model_config = model.config.to_dict()

    # Save model configs
    checkpoint_dir = setup_utils.get_checkpoint_dir(config)
    checkpoint_savepath = os.path.join(checkpoint_dir, 'model_config.yaml')
    write_yaml_to_file(model_config, checkpoint_savepath)

    # Copy to output
    output_dir = setup_utils.get_output_dir(config)
    output_savepath = os.path.join(output_dir, 'model_config.yaml')
    shutil.copy(checkpoint_savepath, output_savepath)


def save_model(model, update_nb, config):
    checkpoint_dir = setup_utils.get_checkpoint_dir(config)
    output_dir = setup_utils.get_output_dir(config)

    # Save the weights
    weight_savepath = os.path.join(checkpoint_dir, 'checkpoint_{}.h5')
    weight_savepath = weight_savepath.format(update_nb)
    model.save_weights(weight_savepath)

    # Copy to output
    output_savepath = os.path.join(output_dir, 'checkpoint.h5')
    shutil.copy(weight_savepath, output_savepath)


def log_metrics(update_nb, obj, log_with_logger=False):
    for k, v in obj.items():
        if isinstance(v, tf.Tensor):
            tf.summary.scalar(k, v, step=update_nb)

    if log_with_logger:
        cpu_obj = {'update': update_nb}
        for k, v in obj.items():
            if isinstance(v, tf.Tensor):
                cpu_obj[k] = v.numpy()
            else:
                cpu_obj[k] = v
        sys.stdout.write('\r')
        logger.info('{}'.format(cpu_obj))


def do_train(config):
    logger.info('Running {}'.format(os.path.basename(__file__)))

    # Load dataset iterators
    logger.info('Loading dataset')
    dataset, valid_dataset = load_datasets(config)
    dataset_iter = make_tf_dataset_iterator(dataset, config, True)

    # Load model and optimizer
    logger.info('Making model')
    train_dtype = tf.float16 if config.train_configs.fp16 else tf.float32
    if config.train_configs.fp16:
        warnings.warn(
            'Mixed precision training is not yet supported on tf2.0. '
            'Switching back to single precision training.'
        )
        train_dtype = tf.float32
    model, optimizer = load_model_and_optimizer(dataset, train_dtype, config)

    # Define some shared objects and training constants
    log_dir = setup_utils.get_log_dir(config)
    writer = tf.summary.create_file_writer(log_dir)
    tgt_pad_idx = dataset.tgt_spm_model.pad_id()
    dummy_new_state_order = tf.constant([], dtype=train_dtype)

    total_num_steps = config.train_configs.num_steps
    total_num_updates = total_num_steps // config.train_configs.update_freq
    shared_obj = {
        'prev_log_time': 0,
        'prev_save_time': 0
    }

    # Define step and update functions
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ])
    def take_one_step(src_tokens, tgt_tokens, src_lengths):
        # Get input and target labels
        bsz = tf.cast(tf.shape(src_tokens)[0], dtype=train_dtype)
        prev_output_tokens = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        # Forward
        logits = model((
            src_tokens,
            src_lengths,
            prev_output_tokens,
            dummy_new_state_order
        ), training=True)

        # Remove padding
        keep_pos = labels != tgt_pad_idx
        labels = labels[keep_pos]
        logits = logits[keep_pos]

        loss, nll_loss = label_smoothed_nll_loss(labels, logits)

        return loss, nll_loss, bsz

    @tf.function
    def _do_update():
        # Init placeholders for training metrics
        total_loss = tf.constant(0, dtype=train_dtype)
        total_nll_loss = tf.constant(0, dtype=train_dtype)
        total_bsz = tf.constant(0, dtype=train_dtype)

        for i in tf.range(config.train_configs.update_freq):
            with tf.GradientTape() as tape:
                src_tokens, tgt_tokens, src_lengths, _ = next(dataset_iter)
                loss, nll_loss, bsz = \
                    take_one_step(src_tokens, tgt_tokens, src_lengths)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss
            total_nll_loss += nll_loss
            total_bsz += bsz

        return (
            total_loss / config.train_configs.update_freq,
            total_nll_loss / config.train_configs.update_freq,
            total_bsz
        )

    def do_update(update_nb):
        """
        Wrapper around _do_update to
        update learning rate and do logging/checkpointing
        """

        # Update learning rate
        optimizer.learning_rate = compute_learning_rate(update_nb, config)

        if config.train_configs.update_freq > 1:
            warnings.warn(
                'Update freq > 1 currently not working, waiting for tf2.0 '
                'update to allow for proper gradient accumulation.'
            )

        # Do backprop
        loss, nll_loss, bsz = _do_update()

        # Do logging
        metrics = {'bsz': bsz, 'loss': loss, 'nll_loss': nll_loss}
        cur_time = time()
        log_interval = config.log_configs.log_interval
        if cur_time >= shared_obj['prev_log_time'] + log_interval:
            log_metrics(update_nb, metrics, True)
            shared_obj['prev_log_time'] = cur_time
        else:
            log_metrics(update_nb, metrics, False)
        writer.flush()

        # Save model
        checkpoint_interval = config.log_configs.checkpoint_interval
        if cur_time >= shared_obj['prev_save_time'] + checkpoint_interval:
            save_model(model, update_nb, config)
            do_validation(update_nb)
            shared_obj['prev_save_time'] = cur_time

    def do_validation(update_nb):
        warnings.warn(
            'Validation currently not done due to already high training GPU '
            'memory requirements. TODO: Find root cause of issue.'
        )
        return

        sys.stdout.write('\r')
        logger.info('Doing validation')
        start_time = time()
        valid_dataset_iter = make_dataset_generator(valid_dataset, config)
        total_loss = 0
        total_nll_loss = 0
        total_bsz = 0
        for src_tokens, tgt_tokens, src_lengths, _ in valid_dataset_iter:
            loss, nll_loss, bsz = take_one_step(
                src_tokens=tf.constant(src_tokens, dtype=tf.int32),
                tgt_tokens=tf.constant(tgt_tokens, dtype=tf.int32),
                src_lengths=tf.constant(src_lengths, dtype=tf.int32)
            )
            bsz = bsz.numpy()
            total_loss += loss.numpy() * bsz
            total_nll_loss += nll_loss.numpy() * bsz
            total_bsz += bsz

        finish_time = time()
        time_taken = finish_time - start_time
        logger.info('Validation completed in {:.1f}s'.format(time_taken))

        log_metrics(update_nb, {
            'valid_loss': total_loss / total_bsz,
            'valid_nll_loss': total_nll_loss / total_bsz
        }, log_with_logger=True)
        writer.flush()

    # Training starts
    logger.info('Ready to begin training')
    start_time = time()

    with writer.as_default():
        # First trace
        do_update(0)
        model.summary()
        save_model_config(model, config)

        # Do training loop
        for i in tqdm(range(1, total_num_updates), desc='Training', ncols=80):
            do_update(i)

        # Save final model and do validation
        save_model(model, total_num_updates, config)
        do_validation(total_num_updates)

    finish_time = time()
    time_taken = finish_time - start_time
    logger.info('Training done in {:.1f}s'.format(time_taken))


def main():
    args = parse_args()
    config = make_config(config_file=args.config_file)
    setup_utils.do_setup(
        config,
        logger_or_name=logger,
        logfile_prefix='train'
    )
    do_train(config)


if __name__ == "__main__":
    main()
