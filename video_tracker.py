import tensorflow as tf
import ssd_model
import matcher
from matcher import Matcher
import coco_loader as coco
import ade_loader
import constants as c
from constants import layer_boxes
from ssd_common import *
import numpy as np
import tf_common as tfc
import signal
import sys
import cv2
import colorsys
import time
import skimage.transform
import skimage.io as io
import webcam
import pickle
import loaderutil
import threading
import trainer

flags = tf.app.flags
FLAGS = flags.FLAGS

def show_video(loader):
    cap = cv2.VideoCapture('test/videoplayback.mp4')

    ssd = ssd_model.SSD()

    cv2.namedWindow("outputs", cv2.WINDOW_NORMAL)

    boxes_ = None
    confidences_ = None

    while cap.isOpened():
        re, sample = cap.read()

        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        resized_img = skimage.transform.resize(sample, (image_size, image_size))

        pred_amax_f, pred_argmax_f, pred_locs_f = ssd.sess.run([ssd.pred_labels_amax, ssd.pred_labels_argmax, ssd.pred_locs],
                                                        feed_dict={ssd.imgs_ph: [resized_img], ssd.bn: False})


        boxes_, confidences_ = matcher.format_output(pred_amax_f[0], pred_argmax_f[0], pred_locs_f[0], boxes_, confidences_)

        loaderutil.resize_boxes(resized_img, sample, boxes_)
        trainer.draw_outputs(np.asarray(sample) / 255.0, boxes_, confidences_, loader.i2name, wait=10)

    cap.release()

if __name__ == "__main__":
    flags.DEFINE_string("model_dir", "summaries/test0", "model directory")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_string("mode", "train", "train, images, image, webcam")

    ade = ade_loader.Pooler(100, FLAGS.batch_size)
    c.classes = 100

    show_video(ade)