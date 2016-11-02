from pycocotools.coco import COCO
import numpy as np
import os
import skimage.io as io
import cv2
import random
import skimage.transform
from constants import image_size, classes
from ssd_common import draw_ann
import threading
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

train_ann_file = "/media/sean/HDCL-UT1/mscoco/annotations/instances_train2014.json"
train_dir = "/media/sean/HDCL-UT1/mscoco/train2014"

val_ann_file = "/media/sean/HDCL-UT1/mscoco/annotations/instances_val2014.json"
val_dir = "/media/sean/HDCL-UT1/mscoco/val2014"

class Loader:
    def __init__(self, train=True):
        if train:
            self.image_dir = train_dir
            ann_file = train_ann_file
            self.get_image_path = self.get_train_path
        else:
            self.image_dir = val_dir
            ann_file = val_ann_file
            self.get_image_path = self.get_val_path

        self.coco = COCO(ann_file)
        cats = self.coco.loadCats(self.coco.getCatIds())
        names = [cat['name'] for cat in cats]
        # id is number from pycocotools
        # i is actual index used
        id2name = dict((cat["id"], cat["name"]) for cat in cats)
        self.id2i = dict((cats[i]['id'], i) for i in range(len(cats)))
        self.i2name = {v: id2name[k] for k, v in self.id2i.iteritems()}
        self.i2name[classes] = "void"

        print("NUMBER OF CLASSES: %i" % len(id2name))

        self.cat_ids = self.coco.getCatIds()
        self.img_ids = self.coco.getImgIds()

        print("%i total training images" % len(self.img_ids))

    def preprocess_batch(self, batch, augment=True):
        imgs = []
        all_used_anns = []

        for img, anns in batch:
            used_anns = []
            w = img.shape[1]
            h = img.shape[0]

            option = np.random.randint(2)

            if not augment:
                option = 0

            if option == 0:
                sample = img
            elif option == 1:
                ratio = random.uniform(0.5, 2.0) #  0.5=portrait, 2.0=landscape
                scale = random.uniform(0.1, 1.0)

                if w > h:
                    p_w = w * scale
                    p_h = p_w / ratio

                    if p_w > w or p_h > h:
                        p_h = h * scale
                        p_w = p_h * ratio
                else:
                    p_h = h * scale
                    p_w = p_h * ratio

                    if p_w > w or p_h > h:
                        p_w = w * scale
                        p_h = p_w / ratio

                if p_w > w or p_h > h:
                    print("error: patch is too big.")

                p_x = random.uniform(0, w - p_w)
                p_y = random.uniform(0, h - p_h)

                sample = img[int(p_y):int(p_y + p_h), int(p_x):int(p_x + p_w)]

                for box, id in anns:
                    box[0] -= p_x
                    box[1] -= p_y

            # warning: this function turns 255 -> 1.0
            resized_img = skimage.transform.resize(sample, (image_size, image_size))

            for box, id in anns:
                scaleX = 1.0 / float(sample.shape[1])
                scaleY = 1.0 / float(sample.shape[0])

                box[0] *= scaleX
                box[1] *= scaleY
                box[2] *= scaleX
                box[3] *= scaleY

            for box, id in anns: # only use boxes with center in image
                cX = box[0] + box[2] / 2.0
                cY = box[1] + box[3] / 2.0

                if cX >= 0 and cX <= 1 and cY >= 0 and cY <= 1:
                    used_anns.append((box, id))

            if random.uniform(0.0, 1.0) < 0.5:
                resized_img = np.fliplr(resized_img)

                for box, id in used_anns:
                    box[0] = 1.0 - box[0] - box[2]

            imgs.append(resized_img)
            all_used_anns.append(used_anns)

        return np.asarray(imgs), all_used_anns

    def get_train_path(self, id):
        return "COCO_train2014_%012d.jpg" % id

    def get_val_path(self, id):
        return "COCO_val2014_%012d.jpg" % id

    def create_batches(self, batch_size, shuffle=True):
        # 1 batch = [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]
        batch = []

        while True:
            indices = range(len(self.img_ids))

            if shuffle:
                indices = np.random.permutation(indices)

            for index in indices:
                img = self.coco.loadImgs(self.img_ids[index])[0]
                path = os.path.join(self.image_dir, self.get_image_path(img['id']))
                I = io.imread(path)

                if len(I.shape) != 3:
                    continue

                ann_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.cat_ids, iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                ann_list = []

                for ann in anns:
                    bb = [f for f in ann["bbox"]]
                    ann_list.append((bb, self.id2i[ann["category_id"]]))

                batch.append((I, ann_list))

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

class PoolLoader:
    def __init__(self):
        self.loader = Loader(True)
        self.batches = self.loader.create_batches(FLAGS.batch_size)
        self.pool = []

    def fill_pool(self):
        while True:
            if len(self.pool) < 100:
                self.pool.append(self.batches.next())

    def start(self):
        t = threading.Thread(target=self.fill_pool)
        t.start()

    def get_batch(self):
        return self.pool.pop(0)

if __name__ == "__main__":
    loader = Loader()
    batch = loader.create_batches(1, shuffle=False)

    for b in batch:
        # [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]

        imgs, anns = loader.preprocess_batch(b)

        I = imgs[0] * 255.0

        for box_coords, id in anns[0]:
            draw_ann(I, box_coords, loader.i2name[id])

        I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("original image", cv2.cvtColor(b[0][0], cv2.COLOR_RGB2BGR))
        cv2.imshow("patch", I)
        if cv2.waitKey(0) == 27:
            break



