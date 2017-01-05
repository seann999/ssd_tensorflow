import random
import skimage.transform
import numpy as np
import constants as c
from constants import layer_boxes

def preprocess_batch(batch, image_size, augment=True):
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
            ratio = random.uniform(0.5, 2.0)  # 0.5=portrait, 2.0=landscape
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

        for box, id in anns:  # only use boxes with center in image
            cX = box[0] + box[2] / 2.0
            cY = box[1] + box[3] / 2.0

            if cX >= 0 and cX <= 1 and cY >= 0 and cY <= 1:
                # fit box completely inside image
                fit_x = max(0, box[0])
                fit_y = max(0, box[1])
                fit_w = min(1.0 - fit_x, box[2])
                fit_h = min(1.0 - fit_y, box[3])
                fit_box = [fit_x, fit_y, fit_w, fit_h]

                used_anns.append((fit_box, id))

        if augment and random.uniform(0.0, 1.0) < 0.5:
            resized_img = np.fliplr(resized_img)

            for box, id in used_anns:
                box[0] = 1.0 - box[0] - box[2]

        imgs.append(resized_img)
        all_used_anns.append(used_anns)

    try:
        return np.asarray(imgs), all_used_anns
    except:
        print([img.shape for img in imgs])
        exit(0)

def resize_boxes(resized, original, boxes, scale=1.0):
    scale_x = original.shape[1] / float(resized.shape[1]) * scale
    scale_y = original.shape[0] / float(resized.shape[0]) * scale

    for o in range(len(layer_boxes)):
        for y in range(c.out_shapes[o][2]):
            for x in range(c.out_shapes[o][1]):
                for i in range(layer_boxes[o]):
                    boxes[o][x][y][i][0] *= scale_x
                    boxes[o][x][y][i][1] *= scale_y
                    boxes[o][x][y][i][2] *= scale_x
                    boxes[o][x][y][i][3] *= scale_y