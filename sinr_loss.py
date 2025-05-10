import tensorflow as tf


@tf.function
def neg_log(x):
    return -tf.math.log(x + 1e-5)


def sinr_loss(y_true, yhat, yhat_bg, pos_weight, alpha=1.0):
    """
    Custom SINR loss function
    Args:
        y_true: True labels
        yhat: Predictions from the model for 'x'
        yhat_bg: Predictions from the model for bg/fake samples
        pos_weight: Scaling factor for positive examples
        alpha: Weight for bg loss
    """

    loss_pos = neg_log(1.0 - yhat)
    inds = tf.range(tf.shape(y_true)[0], dtype=tf.int64)
    inds = tf.stack([inds, y_true], axis=1)
    pos_preds = tf.gather_nd(yhat, inds)

    newvals = pos_weight * neg_log(pos_preds)
    loss_pos = tf.tensor_scatter_nd_update(loss_pos, [inds], [newvals])

    loss_bg = neg_log(1.0 - yhat_bg)

    return tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_bg)
