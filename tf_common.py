import tensorflow as tf

xi = tf.contrib.layers.xavier_initializer
xic = tf.contrib.layers.xavier_initializer_conv2d

def batch_norm(x, train_phase, decay=0.9, eps=1e-5):
    shape = x.get_shape().as_list()

    assert len(shape) in [2, 4]

    n_out = shape[-1]
    beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0))
    gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.constant_initializer(1))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


def conv2d(name, x, maps_in, maps_out, train_phase, size=3, stride=1, act=tf.nn.relu):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d", shape=[size, size, maps_in, maps_out], initializer=xic())
        c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        #b = tf.get_variable(name="conv2d_b", shape=[maps_out], initializer=tf.constant_initializer(0))
        bn = batch_norm(c, train_phase)
        #bn = b + c

    if act is not None:
        return act(bn, name=name)
    else:
        return bn


def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)