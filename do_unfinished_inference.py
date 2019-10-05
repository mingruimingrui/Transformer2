import io
import warnings
from tqdm import tqdm

import numpy as np
import sentencepiece

import tensorflow as tf
from transformer_2.models import Transformer
from transformer_2.data.processing import make_processor_from_list
from transformer_2.utils.batching import batch_tokenized_sents


MODEL_DTYPE = tf.float16
MAX_SENTS_PER_BATCH = 1
BEAM_SIZE = 1
MAX_POS = 1024

if BEAM_SIZE != 1:
    raise ValueError('Currently beam_size must be 1')
if MAX_SENTS_PER_BATCH != 1:
    raise ValueError('Currently max sents per batch must be 1')


# Make preprocessing functions
processor = make_processor_from_list(['lowercase', 'html_unescape'])
src_spm_model = sentencepiece.SentencePieceProcessor()
src_spm_model.Load('tmp/spm_model.src.model')
tgt_spm_model = sentencepiece.SentencePieceProcessor()
tgt_spm_model.Load('tmp/spm_model.tgt.model')


# Load sentences to translate
with io.open('tmp/news-commentary-v14.de-en.en', 'r', encoding='utf-8') as f:
    en_sents = [processor(l.strip()) for l in f][:100]
    # en_sents = ['hello world!', 'hello']
en_tokenized_sents = []
for sent in tqdm(en_sents, desc='Tokenizing sentences'):
    en_tokenized_sents.append(src_spm_model.encode_as_ids(sent) + [2])


# Form sentences into batches
print('Batching')
_, batches, _, _ = batch_tokenized_sents(
    tokenized_sents=en_tokenized_sents,
    padding_idx=1,
    max_positions=MAX_POS,
    max_batch_tokens=30000,
    max_batch_sentences=MAX_SENTS_PER_BATCH,
    shuffle=True,
    do_optimal_batching=True
)


# Form dataset
def batch_generator():
    for batch in batches:
        yield batch


dataset = tf.data.Dataset.from_generator(
    generator=batch_generator,
    output_types=(tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]))
).prefetch(10)
dataset_iterator = iter(dataset)


# Load model with weights
model = Transformer.make_model(config_dict={
    'encoder_vocab_size': 12000,
    'encoder_padding_idx': 1,
    'decoder_vocab_size': 12000,
    'decoder_padding_idx': 1,
    'normalize_before': False
}, dtype=MODEL_DTYPE)
model((
    tf.constant([[0]], dtype=tf.int32),
    tf.constant([1], dtype=tf.int32),
    tf.constant([[2]], dtype=tf.int32),
    tf.constant([], dtype=tf.int32)
))
model.load_weights('tmp/tf_weights_2.h5')


# Make placeholders
_pred_tokens_init_value = \
    np.zeros([MAX_SENTS_PER_BATCH * BEAM_SIZE, MAX_POS], dtype=np.int32)
_pred_tokens_init_value[:, 0] = 2  # First token is </s> in fairseq
_pred_tokens_init_value = tf.constant(_pred_tokens_init_value)
_scores_init_value = \
    tf.zeros([MAX_SENTS_PER_BATCH * BEAM_SIZE, MAX_POS], dtype=MODEL_DTYPE)
_tgt_lens_init_value = tf.ones([MAX_SENTS_PER_BATCH], dtype=tf.int32)
_max_scores_init_value = tf.zeros_like(_tgt_lens_init_value, dtype=MODEL_DTYPE)

pred_tokens_placeholder = tf.Variable(_pred_tokens_init_value)
scores_placeholder = tf.Variable(_scores_init_value)
tgt_lens_placeholder = tf.Variable(_tgt_lens_init_value)
max_scores_placeholder = tf.Variable(_max_scores_init_value)


# Define generate function
# @tf.function
def translate():
    # Initialize placeholders
    model.decoder.clear_cached_states()
    pred_tokens_placeholder.assign(_pred_tokens_init_value)
    scores_placeholder.assign(_scores_init_value)
    tgt_lens_placeholder.assign(_tgt_lens_init_value)
    max_scores_placeholder.assign(_max_scores_init_value)

    # Get next batch and statistics
    src_tokens, src_lengths = next(dataset_iterator)
    src_tokens_shape = tf.shape(src_tokens)
    batch_size = src_tokens_shape[0]
    src_len = src_tokens_shape[1]
    beammed_batch_size = batch_size * BEAM_SIZE
    max_tgt_len = src_len * 2

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
    pred_tokens_placeholder[:beammed_batch_size, 1].assign(
        tf.reshape(preds, [-1]))
    scores_placeholder[:beammed_batch_size, 0].assign(
        tf.reshape(lprobs, [-1]))
    max_scores_placeholder[:batch_size].assign(lprobs[:, 0])

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
    # Portion that doesn't work for beam_size > 1 and max_batch_sentences > 1
    # Start
    ###########################################################################
    warnings.warn(
        'Currently beam search is not working properly. '
        'It would terminate at the max_tgt_len or when the first EOS token is '
        'predicted.'
    )

    for i in tf.range(2, MAX_POS):
        if tf.greater_equal(i, max_tgt_len):
            break

        # TODO: New state order needs ot be determined
        new_state_order = tf.range(1, dtype=tf.int32)

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

        # Update placeholders
        # TODO: Update should do sliced update such that only active beams
        # are being updated
        # Currently update is only for beam_size = 1, sent_per_batch = 1
        pred_tokens_placeholder[0, i].assign(preds[0, 0])
        scores_placeholder[0, i - 1].assign(lprobs[0, 0])
        max_scores_placeholder.assign_add(lprobs[0])

        # TODO: This part is simply a hack that terminates when EOS
        # token is predicted, it should not be in the final code
        if tf.equal(preds[0, 0], 2):
            break

    ###########################################################################
    # Portion that doesn't work for beam_size > 1 and max_batch_sentences > 1
    # End
    ###########################################################################

    return batch_size, max_tgt_len, src_tokens


# Do translation
with tqdm(total=len(en_sents), desc='Translating') as pbar:
    for i in range(len(batches)):
        # Do incremental decoding
        batch_size, max_tgt_len, src_tokens = translate()
        pbar.update(batch_size.numpy())

        # # Print src sentence
        # print('\nSource sentences')
        # for i, token_ids in enumerate(src_tokens.numpy()):
        #     print(i, src_spm_model.decode_ids(token_ids.tolist()))

        # # Retrieve value from placeholders
        # beammed_batch_size = batch_size * BEAM_SIZE
        # pred_tokens = \
        #     pred_tokens_placeholder[:beammed_batch_size, :max_tgt_len]
        # # pred_tokens = pred_tokens[::BEAM_SIZE]
        # scores = scores_placeholder[:beammed_batch_size, max_tgt_len]
        # # scores = scores[::BEAM_SIZE]
        # max_scores = max_scores_placeholder[:batch_size]

        # # Print translation
        # print('\nTranslations')
        # for i, token_ids in enumerate(pred_tokens.numpy()):
        #     print(i, tgt_spm_model.decode_ids(token_ids.tolist()))

print('Done')
