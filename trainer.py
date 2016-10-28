import tensorflow as tf
import model
import matcher
from matcher import Matcher
import coco_loader as coco
import constants
from constants import layer_boxes, draw_rect, classes, image_size, batch_size, center2cornerbox, corner2centerbox
import numpy as np
import tf_common as tfc
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS

np.set_printoptions(threshold=np.nan)

def default2global(default, offsets):
    cX = default[0] + offsets[0]
    cY = default[1] + offsets[1]
    w = default[2] + offsets[2]
    h = default[3] + offsets[3]

    return [cX - w/2.0, cY - h/2.0, w, h]

def calc_offsets(default, truth):
    return [truth[0] - default[0],
            truth[1] - default[1],
            truth[2] - default[2],
            truth[3] - default[3]]

def init_indices2index(out_shapes):
    index = 0
    indices = [[[[None for i in range(layer_boxes[o])] for y in range(out_shapes[o][2])] for x in
                range(out_shapes[o][1])]
               for o in range(len(layer_boxes))]

    for o in range(len(layer_boxes)):
        for y in range(out_shapes[o][2]):
            for x in range(out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    indices[o][y][x][i] = index
                    index += 1

    return indices

def prepare_feed(matches, out_shapes, total_boxes, defaults):
    positives_list = []
    posandnegs_list = []
    true_labels_list = []
    true_locs_list = []

    for o in range(len(layer_boxes)):
        for y in range(out_shapes[o][2]):
            for x in range(out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    match = matches[o][x][y][i]

                    if isinstance(match, tuple):
                        positives_list.append(1)
                        posandnegs_list.append(1)
                        true_labels_list.append(match[1]) #id
                        default = defaults[o][x][y][i]
                        true_locs_list.append(calc_offsets(default, corner2centerbox(match[0])))
                    elif match == -1:
                        positives_list.append(0)
                        posandnegs_list.append(1)
                        true_labels_list.append(classes - 1) # background class
                        true_locs_list.append([0]*4)
                    else:
                        positives_list.append(0)
                        posandnegs_list.append(0)
                        true_labels_list.append(classes - 1)  # background class
                        true_locs_list.append([0]*4)

    re_positives = np.asarray(positives_list)
    re_posandnegs = np.asarray(posandnegs_list)
    re_true_labels = np.asarray(true_labels_list)
    re_true_locs = np.asarray(true_locs_list)

    return re_positives, re_posandnegs, re_true_labels, re_true_locs

def draw_matches(I, out_shapes, boxes, matches, anns):
    I = np.copy(I)

    for o in range(len(layer_boxes)):
        for y in range(out_shapes[o][2]):
            for x in range(out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    s = matches[o][x][y][i]

                    # None if not positive nor negative
                    # -1 if negative
                    # ground truth indices if positive

                    if s == -1:
                        coords = center2cornerbox(boxes[o][x][y][i])
                        draw_rect(I, coords, (255, 0, 0))
                    elif isinstance(s, tuple):
                        coords = center2cornerbox(boxes[o][x][y][i])
                        draw_rect(I, coords, (0, 0, 255))
                        # elif s == 2:
                        #    draw_rect(I, boxes[o][x][y][i], (0, 0, 255), 2)

    for gtbox, id in anns:
        draw_rect(I, gtbox, (0, 255, 0), 3)
        cv2.putText(I, coco.i2name[id], (int(gtbox[0] * image_size), int((gtbox[1] + gtbox[3]) * image_size)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("matches", I)
    cv2.waitKey(1)

def draw_matches2(I, pos, posandneg, true_labels, true_locs, pred_locs, defaults):
    I = np.copy(I)
    index = 0

    for o in range(len(layer_boxes)):
        for y in range(constants.out_shapes[o][2]):
            for x in range(constants.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    if posandneg[index] > 0 and sum(abs(pred_locs[index])) < 100:
                        if pos[index] > 0:
                            d = defaults[o][x][y][i]
                            coords = default2global(d, true_locs[index])
                            draw_rect(I, coords, (0, 255, 0))
                            coords = default2global(d, pred_locs[index])
                            draw_rect(I, coords, (0, 0, 255))
                            cv2.putText(I, coco.i2name[true_labels[index]],
                                        (int(coords[0] * image_size), int((coords[1] + coords[3]) * image_size)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                        else:
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

def draw_outputs(I, boxes, confidences):
    I = np.copy(I)

    for box, conf, top_label in confidences:
        print("%f: %s %s" % (conf, coco.i2name[top_label], box))
        if conf >= 0.1:
            if top_label != classes-1:
                coords = boxes[box[0]][box[1]][box[2]][box[3]]
                coords = center2cornerbox(coords)
                print(coords)

                if abs(sum(coords)) < 100:
                    draw_rect(I, coords, (0, 0, 255))
                    cv2.putText(I, coco.i2name[top_label], (int((coords[0]) * image_size),
                                                            int((coords[1] + coords[3]) * image_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        else: # confidences sorted
            break

    I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("outputs", I)
    cv2.waitKey(1)

def start_train():
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7))))
    imgs_ph, bn, output_tensors, pred_labels, pred_locs = model.model(sess)
    total_boxes = pred_labels.get_shape().as_list()[1]
    positives_ph, posandnegs_ph, true_labels_ph, true_locs_ph, total_loss, class_loss, loc_loss =\
        model.loss(pred_labels, pred_locs, total_boxes)
    out_shapes = [out.get_shape().as_list() for out in output_tensors]
    constants.out_shapes = out_shapes
    constants.indices2index = init_indices2index(out_shapes)

    defaults = model.default_boxes(out_shapes)
    box_matcher = Matcher(out_shapes, defaults)
    batches = coco.create_batches(batch_size, shuffle=True)

    # variables in model are already initialized

    with tf.variable_scope("optimizer"):
        global_step = tf.Variable(0)

        optimizer = tf.train.MomentumOptimizer(1e-3, 0.99).minimize(total_loss, global_step=global_step)
    new_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="optimizer")
    sess.run(tf.initialize_variables(new_vars))

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored %s" % ckpt.model_checkpoint_path)

    while True:
        batch = batches.next()

        imgs, anns = coco.prepare_batch(batch)

        pred_labels_f, pred_locs_f = sess.run([pred_labels, pred_locs], feed_dict={imgs_ph: imgs, bn: False})
        imgs *= 255.0

        batch_values = []

        for batch_i in range(batch_size):
            boxes, matches = box_matcher.match_boxes(pred_labels_f[batch_i], pred_locs_f[batch_i], anns, batch_i)

            positives_f, posandnegs_f, true_labels_f, true_locs_f = prepare_feed(matches, out_shapes, total_boxes, defaults)
            batch_values.append((positives_f, posandnegs_f, true_labels_f, true_locs_f))

            if batch_i == 0:
                b_, c_ = matcher.format_output(pred_labels_f[batch_i], pred_locs_f[batch_i], out_shapes, defaults, batch_i)
                draw_outputs(imgs[batch_i], b_, c_)
                draw_matches(imgs[batch_i], out_shapes, boxes, matches, anns[batch_i])
                draw_matches2(imgs[batch_i], positives_f, posandnegs_f, true_labels_f, true_locs_f, pred_locs_f[batch_i], defaults)

        positives_f, posandnegs_f, true_labels_f, true_locs_f = [np.stack(m) for m in zip(*batch_values)]

        print(positives_f.shape)
        print(posandnegs_f.shape)
        print(true_labels_f.shape)
        print(pred_labels_f.shape)
        print(true_locs_f.shape)
        print(pred_locs_f.shape)

        _, loss_f, step = sess.run([optimizer, total_loss, global_step], feed_dict={imgs_ph: imgs, bn: True, positives_ph:positives_f, posandnegs_ph:posandnegs_f,
                                           true_labels_ph:true_labels_f, true_locs_ph:true_locs_f})

        print("%i: %f" % (step, loss_f))

        tfc.summary_float(step, "loss", loss_f, summary_writer)

        if step % 100 == 0:
            saver.save(sess, "%s/ckpt" % FLAGS.model_dir, step)

if __name__ == "__main__":
    flags.DEFINE_string("model_dir", "summaries/test0", "model directory")

    start_train()