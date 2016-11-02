import tensorflow as tf
import model
import matcher
from matcher import Matcher
import coco_loader as coco
import constants as c
from constants import layer_boxes, classes
from ssd_common import *
import numpy as np
import tf_common as tfc
import signal
import sys
import cv2
import colorsys
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
from threading import Thread

i2name = None

def default2cornerbox(default, offsets):
    c_x = default[0] + offsets[0]
    c_y = default[1] + offsets[1]
    w = default[2] + offsets[2]
    h = default[3] + offsets[3]

    return [c_x - w/2.0, c_y - h/2.0, w, h]

def calc_offsets(default, truth):
    return [truth[0] - default[0],
            truth[1] - default[1],
            truth[2] - default[2],
            truth[3] - default[3]]

def prepare_feed(matches):
    positives_list = []
    negatives_list = []
    true_labels_list = []
    true_locs_list = []

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    match = matches[o][x][y][i]

                    if isinstance(match, tuple): # there is a ground truth assigned to this default box
                        positives_list.append(1)
                        negatives_list.append(0)
                        true_labels_list.append(match[1]) #id
                        default = c.defaults[o][x][y][i]
                        true_locs_list.append(calc_offsets(default, corner2centerbox(match[0])))
                    elif match == -1: # this default box was chosen to be a negative
                        positives_list.append(0)
                        negatives_list.append(1)
                        true_labels_list.append(classes) # background class
                        true_locs_list.append([0]*4)
                    else: # no influence for this training step
                        positives_list.append(0)
                        negatives_list.append(0)
                        true_labels_list.append(classes)  # background class
                        true_locs_list.append([0]*4)

    a_positives = np.asarray(positives_list)
    a_negatives = np.asarray(negatives_list)
    a_true_labels = np.asarray(true_labels_list)
    a_true_locs = np.asarray(true_locs_list)

    return a_positives, a_negatives, a_true_labels, a_true_locs

def draw_matches(I, boxes, matches, anns):
    I = np.copy(I) * 255.0

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    match = matches[o][x][y][i]

                    # None if not positive nor negative
                    # -1 if negative
                    # ground truth indices if positive

                    if match == -1:
                        coords = center2cornerbox(boxes[o][x][y][i])
                        draw_rect(I, coords, (255, 0, 0))
                    elif isinstance(match, tuple):
                        coords = center2cornerbox(boxes[o][x][y][i])
                        draw_rect(I, coords, (0, 0, 255))
                        # elif s == 2:
                        #    draw_rect(I, boxes[o][x][y][i], (0, 0, 255), 2)

    for gt_box, id in anns:
        draw_rect(I, gt_box, (0, 255, 0), 3)
        cv2.putText(I, i2name[id], (int(gt_box[0] * image_size), int((gt_box[1] + gt_box[3]) * image_size)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("matches", I)
    cv2.waitKey(1)

def draw_matches2(I, pos, neg, true_labels, true_locs):
    I = np.copy(I) * 255.0
    index = 0

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    if pos[index] > 0:
                        d = c.defaults[o][x][y][i]
                        coords = default2cornerbox(d, true_locs[index])
                        draw_rect(I, coords, (0, 255, 0))
                        coords = center2cornerbox(d)
                        draw_rect(I, coords, (0, 0, 255))
                        cv2.putText(I, i2name[true_labels[index]],
                                    (int(coords[0] * image_size), int((coords[1] + coords[3]) * image_size)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    elif neg[index] > 0:
                        pass
                        #d = defaults[o][x][y][i]
                        #coords = default2global(d, pred_locs[index])
                        #draw_rect(I, coords, (255, 0, 0))
                        #cv2.putText(I, coco.i2name[true_labels[index]],
                        #            (int(coords[0] * image_size), int((coords[1] + coords[3]) * image_size)),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                    index += 1

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("matches2", I)
    cv2.waitKey(1)

def basic_nms(boxes, confidences, thres=0.45):
    re = []

    def pass_nms(c, lab):
        for box_, conf_, top_label_ in re:
            if lab == top_label_ and calc_jaccard(c, center2cornerbox(boxes[box_[0]][box_[1]][box_[2]][box_[3]])) > thres:
                return False
        return True

    for box, conf, top_label in confidences:
        coords = center2cornerbox(boxes[box[0]][box[1]][box[2]][box[3]])

        if top_label != classes and pass_nms(coords, top_label):
            re.append((box, conf, top_label))

            if len(re) >= 200:
                break

    return re

def draw_outputs(I, boxes, confidences, wait=1):
    I = np.copy(I) * 255.0

    filtered_boxes = []
    filtered = []

    for box, conf, top_label in confidences:
        if conf >= 0.01:
            b = center2cornerbox(boxes[box[0]][box[1]][box[2]][box[3]])
            filtered_boxes.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])
            filtered.append((box, conf, top_label))

    #nms = non_max_suppression_fast(np.asarray(filtered_boxes), 1.00)
    confidences = basic_nms(boxes, filtered)

    for box, conf, top_label in confidences[::-1]:#[filtered[i] for i in nms]:
        if top_label != classes:
            #print("%f: %s %s" % (conf, coco.i2name[top_label], box))
            coords = boxes[box[0]][box[1]][box[2]][box[3]]
            coords = center2cornerbox(coords)
            c = colorsys.hsv_to_rgb(((top_label * 17) % 255) / 255.0, 1.0, 1.0)
            c = tuple([255*c[i] for i in range(3)])

            draw_ann(I, coords, i2name[top_label], color=c, confidence=conf)

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("outputs", I)
    cv2.waitKey(wait)

def start_train(train=True):
    sess = tf.Session()
    imgs_ph, bn, output_tensors, pred_labels, pred_locs = model.model(sess)
    total_boxes = pred_labels.get_shape().as_list()[1]
    positives_ph, negatives_ph, true_labels_ph, true_locs_ph, total_loss, class_loss, loc_loss =\
        model.loss(pred_labels, pred_locs, total_boxes)
    out_shapes = [out.get_shape().as_list() for out in output_tensors]
    c.out_shapes = out_shapes
    c.defaults = model.default_boxes(out_shapes)
    box_matcher = Matcher()

    # variables in model are already initialized, so only initialize those declared after
    with tf.variable_scope("optimizer"):
        global_step = tf.Variable(0)
        lr_ph = tf.placeholder(tf.float32, shape=[])

        optimizer = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)
    new_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="optimizer")
    sess.run(tf.initialize_variables(new_vars))

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored %s" % ckpt.model_checkpoint_path)

    t = time.time()

    if train:
        def signal_handler(signal, frame):
            print('You pressed Ctrl+C!')
            saver.save(sess, "%s/ckpt" % FLAGS.model_dir, step)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        #train_loader = coco.Loader(True)
        global i2name
        #i2name = train_loader.i2name
        #train_batches = train_loader.create_batches(FLAGS.batch_size, shuffle=True)
        train_loader = coco.PoolLoader()
        i2name = train_loader.loader.i2name

        while True:
            #batch = train_batches.next()
            batch = train_loader.get_batch()

            imgs, anns = train_loader.preprocess_batch(batch)

            pred_labels_f, pred_locs_f, step = sess.run([pred_labels, pred_locs, global_step], feed_dict={imgs_ph: imgs, bn: False})

            batch_values = [None for i in range(FLAGS.batch_size)]

            def match_boxes(batch_i):
                #a = time.time()
                matches = box_matcher.match_boxes(pred_labels_f[batch_i], anns[batch_i])
                #print("a: %f" % (time.time() - a))
                #a = time.time()
                positives_f, negatives_f, true_labels_f, true_locs_f = prepare_feed(matches)

                batch_values[batch_i] = (positives_f, negatives_f, true_labels_f, true_locs_f)

                if batch_i == 0:
                    boxes_, confidences_ = matcher.format_output(pred_labels_f[batch_i], pred_locs_f[batch_i])
                    if FLAGS.display:
                        draw_outputs(imgs[batch_i], boxes_, confidences_)
                        draw_matches(imgs[batch_i], c.defaults, matches, anns[batch_i])
                        draw_matches2(imgs[batch_i], positives_f, negatives_f, true_labels_f, true_locs_f)
                #print("b: %f" % (time.time() - a))

            for batch_i in range(FLAGS.batch_size):
                match_boxes(batch_i)

            positives_f, negatives_f, true_labels_f, true_locs_f = [np.stack(m) for m in zip(*batch_values)]

            if step < 4000:
                lr = 8e-4
            elif step < 180000:
                lr = 1e-3
            elif step < 240000:
                lr = 1e-4
            else:
                lr = 1e-5

            _, c_loss_f, l_loss_f, loss_f, step = sess.run([optimizer, class_loss, loc_loss, total_loss, global_step],
                                       feed_dict={imgs_ph: imgs, bn: True, positives_ph:positives_f, negatives_ph:negatives_f,
                                               true_labels_ph:true_labels_f, true_locs_ph:true_locs_f, lr_ph:lr})

            t = time.time() - t
            print("%i: %f (%f secs)" % (step, loss_f, t))
            t = time.time()

            tfc.summary_float(step, "loss", loss_f, summary_writer)
            tfc.summary_float(step, "class loss", c_loss_f, summary_writer)
            tfc.summary_float(step, "loc loss", l_loss_f, summary_writer)

            if step % 1000 == 0:
                saver.save(sess, "%s/ckpt" % FLAGS.model_dir, step)
    else:
        cv2.namedWindow("outputs", cv2.WINDOW_NORMAL)
        print("DETECTION ON TEST IMAGES")
        loader = coco.Loader(False)
        test_batches = loader.create_batches(1, shuffle=True)
        global i2name
        i2name = loader.i2name

        while True:
            batch = test_batches.next()
            imgs, anns = loader.preprocess_batch(batch)
            pred_labels_f, pred_locs_f, step = sess.run([pred_labels, pred_locs, global_step],
                                                        feed_dict={imgs_ph: imgs, bn: False})
            boxes_, confidences_ = matcher.format_output(pred_labels_f[0], pred_locs_f[0])
            draw_outputs(imgs[0], boxes_, confidences_, wait=0)

if __name__ == "__main__":
    flags.DEFINE_string("model_dir", "summaries/test0", "model directory")
    flags.DEFINE_integer("batch_size", 32, "batch size")
    flags.DEFINE_boolean("display", True, "display relevant windows")
    flags.DEFINE_boolean("test", False, "test")

    start_train(train=not FLAGS.test)
