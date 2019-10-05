#!/usr/bin/env python

import os
import sys
import time
from tqdm import tqdm

import tensorflow as tf

from transformer_2.data.datasets import TranslationDataset
from transformer_2.models import Transformer
from transformer_2.losses.label_smoothed_nll_loss \
    import label_smoothed_nll_loss
from transformer_2.utils.io_utils import write_yaml_to_file

TF_DTYPE = tf.float32  # fp16 training is not working well
# Issue with numerical stability (likely in softmax)


# Load dataset
if False:
    dataset = TranslationDataset.new(
        cache_dir='debug/data/bin',
        src_corpus_path='debug/data/token.src',
        tgt_corpus_path='debug/data/token.tgt',
        src_spm_model_path='debug/output/spm_model.src.model',
        tgt_spm_model_path='debug/output/spm_model.tgt.model',
        corpus_format='piece',
        num_workers=30
    )
else:
    dataset = TranslationDataset('debug/data/bin')


# Make tf dataset iterator
tf_dataset = dataset.make_tf_dataset(
    max_batch_tokens=8192,  # 8192, 16384
    max_batch_sentences=512,
    shuffle=True,
    buffer_size=3000000,
    do_optimal_batching=True,
    generate_infinitely=True,
    warn_on_skip=False
).prefetch(10)
tf_dataset_iterator = iter(tf_dataset)


# Load model
model = Transformer.make_model(config_dict={
    'encoder_vocab_size': 8000,
    'encoder_padding_idx': 1,
    'decoder_vocab_size': 8000,
    'decoder_padding_idx': 1,
    'dropout': 0.3,
    'attn_dropout': 0.1,
    'activation_dropout': 0.1
}, dtype=TF_DTYPE)


# Define loss function and optimizer
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric_fn = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-9, beta_2=0.98)


# Define step and update functions
dummy_new_state_order = tf.constant([], dtype=TF_DTYPE)


@tf.function
def take_one_step():
    # Get next batch
    src_tokens, tgt_tokens, src_lengths, _ = next(tf_dataset_iterator)
    bsz = tf.cast(tf.shape(src_tokens)[0], dtype=TF_DTYPE)

    # Get input and target labels
    prev_output_tokens = tgt_tokens[:, :-1]
    labels = tgt_tokens[:, 1:]

    # Forward
    logits = model((
        src_tokens, src_lengths, prev_output_tokens, dummy_new_state_order
    ), training=True)

    # Remove ignore
    keep_pos = labels != 1
    labels = labels[keep_pos]
    logits = logits[keep_pos]

    # Compute loss
    loss, nll_loss = label_smoothed_nll_loss(labels, logits)
    # loss = loss_fn(labels, logits)

    return loss, nll_loss, bsz


@tf.function
def train_eight_steps():
    STEPS_PER_UPDATE = 8

    # Init placeholders for training metrics
    total_loss = tf.constant(0, dtype=TF_DTYPE)
    total_nll_loss = tf.constant(0, dtype=TF_DTYPE)
    total_bsz = tf.constant(0, dtype=TF_DTYPE)

    for i in tf.range(STEPS_PER_UPDATE):
        with tf.GradientTape() as tape:
            loss, nll_loss, bsz = take_one_step()
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += loss
        total_nll_loss += nll_loss
        total_bsz += bsz

    return (
        total_loss / STEPS_PER_UPDATE,
        total_nll_loss / STEPS_PER_UPDATE,
        total_bsz
    )


# Define other utility functions
def save_model(model, update_id, save_config=False):
    checkpoint_path = 'debug/checkpoints'

    if save_config:
        config_savepath = os.path.join(checkpoint_path, 'config.yaml')
        write_yaml_to_file(model.config.to_dict(), config_savepath)

    # Save the weights
    weight_savepath = os.path.join(checkpoint_path, 'checkpoint_{}.h5')
    model.save_weights(weight_savepath.format(update_id))


def log_progress(obj):
    sys.stdout.write('\r')
    print(obj)


# Start tf writer
writer = tf.summary.create_file_writer('tmp')

# First trace
loss, nll_loss, bsz = train_eight_steps()
save_model(model, '0', save_config=True)

# Do training loop
total_updates = 25000
start_time = time.time()
prev_log_time = start_time
prev_save_time = start_time


with writer.as_default():
    with tqdm(total=total_updates - 1, desc='Training', ncols=80) as pbar:
        for i in range(1, total_updates):
            pbar.update(1)

            # Update learning rate
            if i <= 4000:
                optimizer.learning_rate = (1e-4 - 1e-9) / 4000 * i + 1e-9

            # Do backprop
            loss, nll_loss, bsz = train_eight_steps()

            # Do logging and checkpointing
            tf.summary.scalar('loss', loss, step=i)
            tf.summary.scalar('nll_loss', nll_loss, step=i)
            tf.summary.scalar('bsz', bsz, step=i)
            tf.summary.scalar('lr', optimizer.learning_rate, step=i)
            writer.flush()

            cur_time = time.time()
            if cur_time >= prev_log_time + 60:
                log_progress({
                    'update': i,
                    'loss': loss.numpy(),
                    'nll_loss': nll_loss.numpy(),
                    'bsz': bsz.numpy()
                })
                prev_log_time = cur_time

            if cur_time >= prev_save_time + 3600:
                save_model(model, i)
                prev_save_time = cur_time

tf.summary.scalar('loss', loss, step=i)
writer.flush()
log_progress({
    'update': 'final',
    'loss': loss.numpy(),
    'nll_loss': nll_loss.numpy(),
    'bsz': bsz.numpy()
})
save_model(model, 'final')
print('Training done in {:.1f}s'.format(time.time() - start_time))
