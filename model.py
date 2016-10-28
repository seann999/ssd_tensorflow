import numpy as np
import tensorflow as tf
import cv2
import vgg.ssd_base as vgg16
import vgg.utils as utils
import tf_common as tfc
import coco_loader as coco
from constants import image_size, layer_boxes, classes, draw_rect
from matcher import Matcher

ratios = [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0]
conv4_3_ratios = [1.0, 0.5, 2.0]
conv4_3_box_scale = 0.07
box_s_min = 0.1

def model(sess):
    images = tf.placeholder("float", [None, image_size, image_size, 3])
    bn = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    h = [512, 1024, 1024,
         256, 512,
         128, 256,
         128, 256]

    with tf.variable_scope("ssd_extend"):
        c6 = tfc.conv2d("c6", vgg.conv5_3, h[0], h[1], bn, size=3)
        c7 = tfc.conv2d("c7", c6, h[1], h[2], bn, size=1)

        c8_1 = tfc.conv2d("c8_1", c7, h[2], h[3], bn, size=1)
        c8_2 = tfc.conv2d("c8_2", c8_1, h[3], h[4], bn, size=3, stride=2)

        c9_1 = tfc.conv2d("c9_1", c8_2, h[4], h[5], bn, size=1)
        c9_2 = tfc.conv2d("c9_2", c9_1, h[5], h[6], bn, size=3, stride=2)

        c10_1 = tfc.conv2d("c10_1", c9_2, h[6], h[7], bn, size=1)
        c10_2 = tfc.conv2d("c10_2", c10_1, h[7], h[8], bn, size=3, stride=2)

        p11 = tf.nn.avg_pool(c10_2, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")

        out1 = tfc.conv2d("out1", vgg.conv4_3, 512, layer_boxes[0] * (classes+4), bn, size=3, act=None)
        out2 = tfc.conv2d("out2", c7, h[2], layer_boxes[1] * (classes + 4), bn, size=3, act=None)
        out3 = tfc.conv2d("out3", c8_2, h[4], layer_boxes[2] * (classes + 4), bn, size=3, act=None)
        out4 = tfc.conv2d("out4", c9_2, h[6], layer_boxes[3] * (classes + 4), bn, size=3, act=None)
        out5 = tfc.conv2d("out5", c10_2, h[8], layer_boxes[4] * (classes + 4), bn, size=3, act=None)
        out6 = tfc.conv2d("out6", p11, h[8], layer_boxes[5] * (classes + 4), bn, size=1, act=None)

    new_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="ssd_extend")
    sess.run(tf.initialize_variables(new_vars))

    outputs = [out1, out2, out3, out4, out5, out6]

    outfs = []
    for i, out in zip(range(len(outputs)), outputs):
        w = out.get_shape().as_list()[2]
        h = out.get_shape().as_list()[1]
        outf = tf.reshape(out, [-1, w*h*layer_boxes[i], classes+4])
        outfs.append(outf)

    formatted_outs = tf.concat(1, outfs)

    pred_labels = formatted_outs[:,:,:classes]
    pred_locs = formatted_outs[:, :, classes:]

    return images, bn, outputs, pred_labels, pred_locs

def smooth_l1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    re = tf.select(condition, l2, l1)

    return re

def loss(pred_labels, pred_locs, total_boxes):
    # positives = (batches x boxes)
    # pred_labels = (batches x boxes x classes)
    # true_labels = (batches x boxes)
    # pred_locs = (batches x boxes x 4)
    positives = tf.placeholder(tf.float32, [None, total_boxes])
    posandnegs = tf.placeholder(tf.float32, [None, total_boxes])
    true_labels = tf.placeholder(tf.int32, [None, total_boxes])
    true_locs = tf.placeholder(tf.float32, [None, total_boxes, 4])

    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_labels, true_labels) * posandnegs
    loc_loss = tf.reduce_sum(smooth_l1(pred_locs - true_locs), reduction_indices=2) * positives # smooth l1
    total_loss = tf.reduce_mean(tf.reduce_sum(class_loss, reduction_indices=1) / (1e-5 + tf.reduce_sum(posandnegs, reduction_indices=1))\
                 + 1.0 * tf.reduce_sum(loc_loss, reduction_indices=1) / (1e-5 + tf.reduce_sum(positives, reduction_indices=1)))

    return positives, posandnegs, true_labels, true_locs, total_loss, class_loss, loc_loss

def box_scale(k):
    s_min = box_s_min
    s_max = 0.95
    m = 6.0

    s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0) # equation 2

    return s_k

def default_boxes(out_shapes):
    boxes = []

    for o_i in range(len(out_shapes)):
        layer_boxes = []

        o = out_shapes[o_i]
        s_k = box_scale(o_i + 1)
        s_k1 = box_scale(o_i + 2)

        for x in range(o[1]):
            x_boxes = []
            for y in range(o[2]):
                y_boxes = []
                #row = o[0, x, y]
                conv4_3 = o_i == 0

                rs = ratios

                if conv4_3:
                    rs = conv4_3_ratios

                for i in range(len(rs)):
                    #diffs = row[(i + 1) * classes:(i + 1) * classes + 4]

                    if conv4_3:
                        scale = conv4_3_box_scale
                    else:
                        scale = s_k

                        if i == 0:
                            scale = np.sqrt(s_k * s_k1)

                    defaultW = scale * np.sqrt(rs[i])
                    defaultH = scale / np.sqrt(rs[i])
                    w = defaultW# + diffs[2]
                    h = defaultH# + diffs[3]

                    cX = (x + 0.5) / float(o[1])# + diffs[0]
                    cY = (y + 0.5) / float(o[2])# + diffs[1]

                    y_boxes.append([cX, cY, w, h])
                x_boxes.append(y_boxes)
            layer_boxes.append(x_boxes)
        boxes.append(layer_boxes)

    return boxes

if __name__ == "__main__":
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7))))
    imgs_ph, bn, outputs = model(sess)
    out_shapes = [out.get_shape().as_list() for out in outputs]
    defaults = default_boxes(out_shapes)
    box_matcher = Matcher(out_shapes, defaults)
    batches = coco.create_batches(8, shuffle=True)

    for batch in batches:
        imgs, anns = coco.prepare_batch(batch)
        out = sess.run(outputs, feed_dict={imgs_ph: imgs, bn: True})
        imgs *= 255.0

        for batch_i in range(8):
            print(batch_i)

            boxes, score = matcher.match_boxes(out, anns, batch_i)
            I = imgs[batch_i]

            for o in range(len(layer_boxes)):
                for x in range(out_shapes[o][1]):
                    for y in range(out_shapes[o][2]):
                        for i in range(layer_boxes[o]):
                            s = score[o][x][y][i]

                            if s == -1:
                                draw_rect(I, boxes[o][x][y][i], (255, 0, 0))
                            elif s == 1:
                                draw_rect(I, boxes[o][x][y][i], (0, 0, 255))
                            elif s == 2:
                                draw_rect(I, boxes[o][x][y][i], (0, 0, 255), 2)

            for gtbox, id in anns[batch_i]:
                draw_rect(I, gtbox, (0, 255, 0), 3)

            I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow("image", I)
            cv2.waitKey(0)

