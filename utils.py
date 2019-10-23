import tensorflow as tf

def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

def true_positive_rate(y_true, y_pred):
    y_true_bool = tf.equal(tf.round(y_true), 1)
    y_pred_bool = tf.equal(tf.round(y_pred), 1)
    tp = tf.cast(tf.math.count_nonzero(tf.logical_and(y_true_bool, y_pred_bool)), tf.float64)
    p = tf.cast(tf.math.count_nonzero(y_true_bool), tf.float64)
    tpr = tp/p
    return tf.where(tf.math.is_nan(tpr), tf.cast(1.0, tf.float64), tpr)
    # return tp/p

def false_positive_rate(y_true, y_pred):
    y_true = tf.equal(tf.round(y_true), 0)
    y_pred = tf.equal(tf.round(y_pred), 0)
    tn = tf.math.count_nonzero(tf.logical_and(y_true, y_pred))
    n = tf.math.count_nonzero(y_true)
    fpr = 1 - tn/n
    return tf.where(tf.math.is_nan(fpr), tf.cast(0.0, tf.float64), fpr)

def cust_accuracy(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    correct = tf.math.less(diff, 0.5)
    correct = tf.cast(correct, tf.float32)
    return tf.math.reduce_mean(correct)

