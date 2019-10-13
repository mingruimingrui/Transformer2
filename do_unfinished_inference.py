import io
import warnings
from time import time
from tqdm import tqdm

import sentencepiece
import numpy as np
import tensorflow as tf

from transformer_2.models import Transformer
from transformer_2.data.processing import make_processor_from_list
from transformer_2.utils.batching import batch_tokenized_sents
from transformer_2.utils.io_utils import read_yaml_from_file

MODEL_DTYPE = tf.float16
MAX_BATCH_SENTENCES = 256
MAX_BATCH_TOKENS = 30000
BEAM_SIZE = 1
PRINT_TRANSLATIONS = False

if BEAM_SIZE != 1:
    raise ValueError('Currently beam_size must be 1')
warnings.warn('Currently beam search is not working properly.')

# Set memory growth on GPU
for d in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(d, True)


# Load model with weights
print('Loading model')
model = Transformer.make_model(
    'debug/output/model_config.yaml',
    dtype=MODEL_DTYPE
)
model((
    tf.constant([[0]], dtype=tf.int32),  # src_tokens
    tf.constant([1], dtype=tf.int32),  # src_lengths
    tf.constant([[0]], dtype=tf.int32),  # prev_output_tokens
    tf.constant([0], dtype=tf.int32),  # new_state_order
))
model.load_weights('debug/output/checkpoint.h5')


# Make processor and load spm models
print('Loading processors')
src_preprocessing_steps = \
    read_yaml_from_file('debug/output/src_preprocessing_steps.yaml')
processor = make_processor_from_list(src_preprocessing_steps)
src_spm_model = sentencepiece.SentencePieceProcessor()
src_spm_model.Load('debug/output/spm_model.src.model')
tgt_spm_model = sentencepiece.SentencePieceProcessor()
tgt_spm_model.Load('debug/output/spm_model.tgt.model')


# Load sentences to translate
print('Loading sentences')
with io.open('debug/newstest2019-ende.en', 'r', encoding='utf-8') as f:
    src_sents = [processor(l.strip()) for l in f][:10000]
tokenized_src_sents = []
for sent in tqdm(src_sents, desc='Tokenizing sentences'):
    tokenized_src_sents.append(
        [src_spm_model.bos_id()] +
        src_spm_model.encode_as_ids(sent) +
        [src_spm_model.eos_id()]
    )


# Form sentences into batches
print('Batching')
_, batches, _, _ = batch_tokenized_sents(
    tokenized_sents=tokenized_src_sents,
    padding_idx=src_spm_model.pad_id(),
    max_positions=model.config['encoder_max_positions'],
    max_batch_tokens=MAX_BATCH_TOKENS,
    max_batch_sentences=MAX_BATCH_SENTENCES,
    do_optimal_batching=True
)


# Form dataset
def batch_generator():
    for batch in batches:
        yield batch


print('Making dataset')
dataset = tf.data.Dataset.from_generator(
    generator=batch_generator,
    output_types=(tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]))
).prefetch(10)
dataset_iterator = iter(dataset)


# Make placeholder init values
decoder_max_positions = model.config['decoder_max_positions']
_pred_tokens_init_value = np.zeros(
    [MAX_BATCH_SENTENCES * BEAM_SIZE, decoder_max_positions],
    dtype=np.int32
)
_pred_tokens_init_value[:, 0] = tgt_spm_model.bos_id()
_pred_tokens_init_value = tf.constant(_pred_tokens_init_value)
_scores_init_value = tf.zeros(
    [MAX_BATCH_SENTENCES * BEAM_SIZE, decoder_max_positions - 1],
    dtype=MODEL_DTYPE
)
_tgt_lens_init_value = tf.ones([MAX_BATCH_SENTENCES], dtype=tf.int32)
_finished_init_value = \
    tf.zeros([MAX_BATCH_SENTENCES * BEAM_SIZE], dtype=tf.bool)

# Make placeholders
pred_tokens_placeholder = tf.Variable(_pred_tokens_init_value)
scores_placeholder = tf.Variable(_scores_init_value)
tgt_lens_placeholder = tf.Variable(_tgt_lens_init_value)
finished_placeholder = tf.Variable(_finished_init_value)


# Define generate function
@tf.function
def translate_batch():
    # Initialize placeholders
    model.decoder.clear_cached_states()
    pred_tokens_placeholder.assign(_pred_tokens_init_value)
    scores_placeholder.assign(_scores_init_value)
    tgt_lens_placeholder.assign(_tgt_lens_init_value)
    finished_placeholder.assign(_finished_init_value)

    # Get next batch and stats
    src_tokens, src_lengths = next(dataset_iterator)
    src_tokens_shape = tf.shape(src_tokens)
    batch_size = src_tokens_shape[0]
    src_len = src_tokens_shape[1]
    beammed_batch_size = batch_size * BEAM_SIZE
    max_tgt_len = tf.cast(tf.cast(src_len, tf.float16) * 1.4 + 2, tf.int32)
    max_tgt_len = tf.minimum(max_tgt_len, decoder_max_positions)

    # Run encoder
    encoder_out, encoder_padding_mask = \
        model.encoder((src_tokens, src_lengths))

    # First pass of decoder
    logits = model.decoder((
        pred_tokens_placeholder[:batch_size, :1],
        encoder_out, encoder_padding_mask,
        tf.constant([], dtype=tf.int32)
    ))
    logits = logits[:, -1, :]  # We are only interested in preds for next token
    lprobs = tf.math.log(tf.nn.softmax(logits, axis=-1))
    preds = tf.argsort(logits, direction='DESCENDING')

    # Get top preds and probs for each sentence
    preds = preds[:, :BEAM_SIZE]  # batch_size * beam_size
    idxs = tf.reshape(tf.range(batch_size, dtype=preds.dtype), [-1, 1])
    idxs = tf.tile(idxs, [1, BEAM_SIZE])
    lprobs = tf.gather_nd(lprobs, tf.stack([idxs, preds], axis=-1))
    # lprobs.shape == [batch_size, beam_size]

    # Update placeholders
    preds = tf.reshape(preds, [-1])
    lprobs = tf.reshape(lprobs, [-1])
    pred_tokens_placeholder[:beammed_batch_size, 1].assign(preds)
    scores_placeholder[:beammed_batch_size, 0].assign(lprobs)
    finished_placeholder[:beammed_batch_size].assign(preds == 2)
    # tgt_len_placeholder already initialized as ones

    # Expand encoder outputs for beam search
    encoder_out = tf.expand_dims(encoder_out, 2)
    encoder_out = tf.tile(encoder_out, [1, 1, BEAM_SIZE, 1])
    encoder_out = tf.reshape(encoder_out, (src_len, beammed_batch_size, -1))

    encoder_padding_mask = tf.expand_dims(encoder_padding_mask, 1)
    encoder_padding_mask = tf.tile(encoder_padding_mask, [1, BEAM_SIZE, 1])
    encoder_padding_mask = tf.reshape(
        encoder_padding_mask,
        (beammed_batch_size, src_len)
    )

    ###########################################################################
    # (START) This for loop doesn't work with beam_size > 1
    ###########################################################################

    for i in tf.range(2, max_tgt_len):
        # TODO: New state order should be determined based on state of
        # beam search
        new_state_order = tf.range(beammed_batch_size, dtype=tf.int32)

        # Compute model predictions
        logits = model.decoder((
            pred_tokens_placeholder[:beammed_batch_size, :i],
            encoder_out, encoder_padding_mask,
            new_state_order
        ))
        logits = logits[:, -1, :]
        lprobs = tf.math.log(tf.nn.softmax(logits, axis=-1))
        preds = tf.argsort(logits, direction='DESCENDING')

        # Get top preds for each sentence
        preds = preds[:, :BEAM_SIZE]  # beammed_batch_size * beam_size
        idxs = tf.reshape(tf.range(batch_size, dtype=preds.dtype), [-1, 1])
        idxs = tf.tile(idxs, [1, BEAM_SIZE])
        lprobs = tf.gather_nd(lprobs, tf.stack([idxs, preds], axis=-1))
        # beammed_batch_size * beam_size

        # Update placeholders
        # TODO: Update should do sliced update such that only active beams
        # are being updated
        # Currently update is only for beam_size = 1
        preds = tf.reshape(preds, [-1])
        lprobs = tf.reshape(lprobs, [-1])

        preds = tf.stack([preds, tf.ones_like(preds) + 1], axis=1)
        lprobs = tf.stack([lprobs, tf.zeros_like(lprobs)], axis=1)

        finished_pos = finished_placeholder[:beammed_batch_size]
        finished_pos = tf.cast(finished_pos, dtype=tf.int32)
        finished_pos = tf.reshape(finished_pos, [-1, 1])
        preds = tf.gather_nd(preds, finished_pos, batch_dims=1)
        lprobs = tf.gather_nd(lprobs, finished_pos, batch_dims=1)

        pred_tokens_placeholder[:beammed_batch_size, i].assign(preds)
        scores_placeholder[:beammed_batch_size, i - 1].assign(lprobs)
        finished_placeholder[:beammed_batch_size].assign(preds == 2)

        # Terminate when all are done
        num_incomplete = finished_placeholder[:beammed_batch_size]
        num_incomplete = 1 - tf.cast(num_incomplete, dtype=tf.int32)
        num_incomplete = tf.reduce_sum(num_incomplete)
        if tf.equal(num_incomplete, 0):
            break

    ###########################################################################
    # (END) This for loop doesn't work with beam_size > 1
    ###########################################################################

    return batch_size, max_tgt_len, src_tokens


# Do translation
batch_size, max_tgt_len, src_tokens = translate_batch()
beammed_batch_size = batch_size * BEAM_SIZE
pred_tokens = pred_tokens_placeholder[:beammed_batch_size, :max_tgt_len]
pred_tokens = pred_tokens[::BEAM_SIZE]
if PRINT_TRANSLATIONS:
    for src_token_ids, tgt_token_ids in zip(
        src_tokens.numpy(), pred_tokens.numpy()
    ):
        print()
        print(src_spm_model.decode_ids(src_token_ids.tolist()))
        print(tgt_spm_model.decode_ids(tgt_token_ids.tolist()))
        print()

with tqdm(total=len(src_sents), desc='Translating') as pbar:
    pbar.update(batch_size.numpy())
    start_time = time()
    num_translated = 0

    for _ in range(len(batches) - 1):
        # Do incremental decoding
        batch_size, max_tgt_len, src_tokens = translate_batch()
        batch_size = batch_size.numpy()
        pbar.update(batch_size)
        num_translated += batch_size

        # Retrieve value from placeholders
        beammed_batch_size = batch_size * BEAM_SIZE
        pred_tokens = \
            pred_tokens_placeholder[:beammed_batch_size, :max_tgt_len]
        pred_tokens = pred_tokens[::BEAM_SIZE]
        scores = scores_placeholder[:beammed_batch_size, :max_tgt_len - 1]
        scores = scores[::BEAM_SIZE]

        if PRINT_TRANSLATIONS:
            for src_token_ids, tgt_token_ids in zip(
                src_tokens.numpy(), pred_tokens.numpy()
            ):
                print()
                print(src_spm_model.decode_ids(src_token_ids.tolist()))
                print(tgt_spm_model.decode_ids(tgt_token_ids.tolist()))
                print()

    finish_time = time()
    time_taken = finish_time - start_time

print('{:.2f} sents/s'.format(num_translated / time_taken))
