import tensorflow as tf


def label_smoothed_nll_loss(
    labels, logits, label_smoothing=0.1,
    ignore_index=None, reduce=True
):
    lprobs = tf.math.log(tf.math.softmax(logits, axis=1))
    eps_i = label_smoothing / tf.cast(tf.shape(lprobs)[-1], logits.dtype)

    labels = tf.expand_dims(labels, 1)
    nll_loss = -tf.gather_nd(lprobs, labels, batch_dims=1)
    smooth_loss = -tf.reduce_sum(lprobs * eps_i, axis=1)

    if ignore_index is not None:
        non_ignore_mask = tf.not_equal(labels, ignore_index)
        nll_loss = nll_loss[non_ignore_mask]
        smooth_loss = smooth_loss[non_ignore_mask]
    else:
        nll_loss = tf.reshape(nll_loss, [-1])
        smooth_loss = tf.reshape(smooth_loss, [-1])

    if reduce:
        nll_loss = tf.reduce_mean(nll_loss)
        smooth_loss = tf.reduce_mean(smooth_loss)

    loss = (1. - label_smoothing) * nll_loss + smooth_loss

    return loss, nll_loss
