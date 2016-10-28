import tensorflow as tf
from tensorflow.python.training import moving_averages

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


def xwb(name, x, _in, out, train_phase):
    with tf.variable_scope(name):
        w = tf.get_variable(name="w", shape=[_in, out], initializer=xi())
        xw = tf.matmul(x, w)
        re = batch_norm(xw, train_phase)

    return re


def conv2d(name, x, maps_in, maps_out, train_phase, size=3, stride=1, act=tf.nn.relu):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d", shape=[size, size, maps_in, maps_out], initializer=xic())
        c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        bn = batch_norm(c, train_phase)

    if act is not None:
        return act(bn, name=name)
    else:
        return bn


def conv2d_t(name, x, maps_in, maps_out, output_size, train_phase, size=3, stride=2, act=tf.nn.relu):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d_t", shape=[size, size, maps_out, maps_in], initializer=xic())
        c = tf.nn.conv2d_transpose(x, w, tf.pack(output_size), strides=[1, stride, stride, 1], padding='SAME')
        #bn = batch_norm(c, train_phase)

    if act is not None:
        return act(c, name=name)
    else:
        return c


def maxpool(name, x):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def load_or_init(model_dir, sess):
    init = tf.initialize_all_variables()

    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored %s" % ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    return saver, summary_writer

def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)