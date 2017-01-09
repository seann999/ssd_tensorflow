import tensorflow as tf

xi = tf.contrib.layers.xavier_initializer
xic = tf.contrib.layers.xavier_initializer_conv2d

def conv2d(name, x, maps_in, maps_out, train_phase, size=3, stride=1, act=tf.nn.elu):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d", shape=[size, size, maps_in, maps_out], initializer=xic())
        c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

        bn = tf.contrib.layers.batch_norm(c,
                                          decay=0.9,
                                          updates_collections=None,
                                          epsilon=1e-5,
                                          scale=True,
                                          is_training=train_phase,
                                          scope="bn")

    if act is not None:
        return act(bn, name=name)
    else:
        return bn

def deconv2d(name, x, maps_in, maps_out, train_phase, output_shape, size=3, stride=1, act=tf.nn.elu, pad="SAME"):
    with tf.variable_scope(name):
        w = tf.get_variable(name="deconv2d", shape=[size, size, maps_out, maps_in], initializer=xic())
        c = tf.nn.conv2d_transpose(x, w, strides=[1, stride, stride, 1], output_shape=output_shape, padding=pad)

        bn = tf.contrib.layers.batch_norm(c,
                                     decay=0.9,
                                     updates_collections=None,
                                     epsilon=1e-5,
                                     scale=True,
                                     is_training=train_phase,
                                     scope="bn")

    if act is not None:
        return act(bn, name=name)
    else:
        return bn


def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)