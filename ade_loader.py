
import os
import numpy as np
import scipy.misc
import operator
import pickle
import cv2
import time

root_root = "/media/sean/HDCL-UT1/ADE20K_2016_07_26/images"
train_root = "/media/sean/HDCL-UT1/ADE20K_2016_07_26/images/training"
valid_root = "/media/sean/HDCL-UT1/ADE20K_2016_07_26/images/validation"

np.random.seed(int(time.time()))

include = []
with open("list.txt") as f:
    for line in f:
        include.append(line.strip())

def gen_bbox(img, id):
    a = np.where(img == id)

    try:
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
        return bbox
    except:
        print(a)
        print("exception at bbox")
        return None

def load_image(path, w2i):
    img = scipy.misc.imread("%s.jpg" % (path))

    if len(img.shape) == 2: # no channel dim
        img = np.dstack([img, img, img])

    seg = scipy.misc.imread("%s_seg.png" % (path))
    B = seg[:, :, 2]
    orig, instances = np.unique(B, return_inverse=True)
    instances = np.reshape(instances, B.shape)
    re = []
    index = 1 # skip first

    with open("%s_atr.txt" % path, "r") as f:
        for line in f.readlines():
            tokens = line.split(" # ")
            instance = int(tokens[0])
            partlvl = int(tokens[1])
            rawname = str(tokens[4])

            if partlvl == 0:
                if rawname in w2i:
                    bb = gen_bbox(instances, index)

                    if bb is None:
                        index += 1
                        continue

                    re.append((w2i[rawname], bb))
                else:
                    #print("%s not in vocab; skipping" % rawname)
                    pass

                index += 1
            else:
                break

    #print("---")

    return img, re

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

def generate_batches(paths, w2i, batch_size=4):
    batch = []

    while True:
        order = np.random.permutation(np.arange(len(paths)))

        for i in order:
            path = paths[i]
            img, boxes = load_image(path, w2i)

            batch.append((img, boxes))

            if len(batch) >= batch_size:
                yield batch
                batch = []

def get_w2i(size):
    with open("ade_counts.p", "r") as f:
        sorted_counts = pickle.loads(f.read())

    #for i in sorted_counts:
    #    print(i)

    w2i = {}
    i2w = []
    i = 0

    while len(i2w) < size:
        if sorted_counts[i][0] in include:
            w2i[sorted_counts[i][0]] = len(i2w)
            i2w.append(sorted_counts[i][0])
        i += 1

    return w2i, i2w

class AdeLoader:
    def __init__(self, classes):
        self.classes = classes
        self.paths = load_paths()
        self.w2i, self.i2name = get_w2i(self.classes)
        print("i2name length: %i" % len(self.i2name))
        self.i2name.append("void")

    def create_batches(self, batch_size, shuffle=True):
        # 1 batch = [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]
        batch = []

        while True:
            order = np.random.permutation(np.arange(len(self.paths)))

            for i in order:
                path = self.paths[i]
                img, rawboxes = load_image(path, self.w2i)

                boxes = []

                for id, box in rawboxes:
                    boxes.append(([box[0], box[1], box[2] - box[0], box[3] - box[1]], id))

                batch.append((img, boxes))

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

if __name__ == "__main__":
    #generate_counts()
    cv2.namedWindow("image")
    paths = load_paths()
    w2i, i2w = get_w2i(100)
    print(i2w)
    for batch in generate_batches(paths, w2i, batch_size=1):
        for id, box in batch[0][1]:
            color = np.random.uniform(0, 255, 3)
            cv2.rectangle(batch[0][0], (box[0], box[1]), (box[2], box[3]), color, 3)
            cv2.putText(batch[0][0], i2w[id], (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("image", cv2.cvtColor(batch[0][0], cv2.COLOR_RGB2BGR))

        cv2.waitKey(0)
