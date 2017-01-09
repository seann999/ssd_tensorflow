
import os
import numpy as np
import scipy.misc
import operator
import pickle
import cv2
import threading
import loaderutil
import time
import constants as c
import skimage.transform

root_root = "/media/sean/HDCL-UT1/ADE20K_2016_07_26/images"
train_root = "/media/sean/HDCL-UT1/ADE20K_2016_07_26/images/training"
valid_root = "/media/sean/HDCL-UT1/ADE20K_2016_07_26/images/validation"

np.random.seed(int(time.time()))

include = []
with open("list.txt") as f:
    for line in f:
        include.append(line.strip())

def load_image(path):
    rawnames = []
    index = 1  # skip first

    with open("%s_atr.txt" % path, "r") as f:
        for line in f.readlines():
            tokens = line.split(" # ")
            instance = int(tokens[0])
            partlvl = int(tokens[1])
            rawname = str(tokens[4])

            if partlvl == 0:
                #if rawname in w2i:
                rawnames.append((rawname, index))
                #else:
                #    pass

                index += 1
            else:
                break

    #if len(rawnames) < 3:
    #    return None, None

    img = scipy.misc.imread("%s.jpg" % (path))

    if len(img.shape) == 2:  # no channel dim
        img = np.dstack([img, img, img])

    seg = scipy.misc.imread("%s_seg.png" % (path))
    B = seg[:, :, 2]
    #orig, instances = np.unique(B, return_inverse=True)
    #instances = np.reshape(instances, B.shape)
    #re = []

    #seg_img = np.zeros([c.image_size, c.image_size, len(w2i)+1])
    #seg_img[:,:,-1] = 1

    #for rawname, index in rawnames:
        #bb, seg = gen_bbox(instances, index)

        #if bb is not None:
            #re.append((w2i[rawname], bb))
            #seg_img[:,:, w2i[rawname]] = 1

    return img, rawnames, B

def generate_counts():
    counts = {}
    w2i = {}
    filesc = 0

    for root, dirs, files in os.walk(root_root):
        for name in files:
            if name.endswith(".txt"):
                path = os.path.join(root, name)
                print(path)

                with open(path, "r") as f:
                    for line in f.readlines():
                        tokens = line.split(" # ")
                        partlvl = int(tokens[1])
                        rawname = str(tokens[4])

                        if partlvl == 0:
                            if rawname in counts:
                                counts[rawname] += 1
                            else:
                                counts[rawname] = 1

                    filesc += 1

    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))[::-1]

    for i in range(len(sorted_counts)):
        w2i[sorted_counts[i][0]] = i

    for i in range(100):
        print("%i: %s" % (i, sorted_counts[i]))

    print(len(sorted_counts))

    with open("counts.p", "w+") as f:
        f.write(pickle.dumps(sorted_counts))


def load_paths(root=train_root):
    paths = []

    for froot, dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jpg"):
                paths.append(os.path.join(froot, name[:-4]))

    return paths

def get_w2i(size, exception=True):
    with open("ade_counts.p", "r") as f:
        sorted_counts = pickle.loads(f.read())

    #for i in sorted_counts:
    #    print(i)

    w2i = {}
    i2w = []
    i = 0

    while len(i2w) < size:
        if (sorted_counts[i][0] in include) or not exception:
            w2i[sorted_counts[i][0]] = len(i2w)
            i2w.append(sorted_counts[i][0])
        i += 1

    return w2i, i2w

class Pooler():
    def __init__(self, classes, batch_size):
        self.classes = classes
        c.classes = classes
        self.batch_size = batch_size
        self.paths = load_paths()
        self.w2i, self.i2name = get_w2i(self.classes)
        self.seg_w2i, self.seg_i2name = get_w2i(self.classes, exception=False)

        print("i2name length: %i" % len(self.i2name))
        self.i2name.append("void")

        self.pool = []
        self.i = 0
        self.order = np.random.permutation(np.arange(len(self.paths)))

        for i in range(8):
            threading._start_new_thread(self.load, (i * 5 + 5,))

    def next_path(self):
        re = self.paths[self.order[self.i]]
        self.i += 1

        if self.i >= len(self.paths):
            self.i = 0
            self.order = np.random.permutation(np.arange(len(self.paths)))

        return re

    def pop(self):
        while len(self.pool) <= 0:
            print("empty")
            time.sleep(1)

        print("pool: %i" % len(self.pool))

        return self.pool.pop(0)

    def load(self, activate):
        while True:
            if len(self.pool) < activate:
                while len(self.pool) < activate+5:
                    batch = []

                    while len(batch) < self.batch_size:
                        path = self.next_path()

                        img, ids, seg_img = load_image(path)

                        if img is None:
                            continue

                        #boxes = []

                        #for id, box in rawboxes:
                        #    boxes.append(([box[0], box[1], box[2] - box[0], box[3] - box[1]], id))

                        batch.append((img, ids, seg_img))

                    imgs, anns, seg_imgs = loaderutil.preprocess_batch2(batch, c.image_size, self.w2i, self.seg_w2i, augment=True)
                    add = [imgs, anns, seg_imgs]
                    self.pool.append(add)
                    #print(len(self.pool))
            else:
                time.sleep(1)


if __name__ == "__main__":
    #generate_counts()
    loader = Pooler(100, 1)
    cv2.namedWindow("image")
    cv2.namedWindow("seg", cv2.WINDOW_NORMAL)

    while True:
        imgs, anns, seg_imgs = loader.pop()

        for box, id in anns[0]:
            color = np.random.uniform(0, 255, 3)
            #cv2.rectangle(imgs[0], (int(box[0]), int(box[1])),
            #              (int((box[2] - box[0])), int((box[3] - box[1]))), color, 3)
            cv2.rectangle(imgs[0], (int(box[0]*c.image_size), int(box[1]*c.image_size)), (int((box[2]+box[0])*c.image_size), int((box[3]+box[1])*c.image_size)), color, 3)
            cv2.putText(imgs[0], loader.i2name[id], (int(box[0]*c.image_size), int(box[1]*c.image_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            print("box")

        #cv2.imshow("seg image", cv2.cvtColor((seg_imgs[0] * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imshow("image", cv2.cvtColor((imgs[0]*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(seg_imgs[0,:,:,0])
        seg = 1.0 - seg_imgs[0,:,:,0]/101.0
        #seg = skimage.transform.resize(seg, (76, 76), order=0)
        cv2.imshow("seg", seg)

        cv2.waitKey(0)
