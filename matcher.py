from constants import layer_boxes, classes, center2cornerbox
import numpy as np

def calc_intersection(r1, r2):
    left = max(r1[0], r2[0])
    right = min(r1[0] + r1[2], r2[0] + r2[2])
    bottom = min(r1[1] + r1[3], r2[1] + r2[3])
    top = max(r1[1], r2[1])

    if left < right and top < bottom:
        return (right - left) * (bottom - top)

    return 0


def calc_jaccard(r1, r2):
    r1_ = [r1[0], r1[1], max(r1[2], 0), max(r1[3], 0)]
    r2_ = [r2[0], r2[1], max(r2[2], 0), max(r2[3], 0)]
    intersection = calc_intersection(r1_, r2_)
    union = r1_[2] * r1_[3] + r2_[2] * r2_[3] - intersection

    if union <= 0:
        return 0

    j = intersection / union

    return j

def format_output(pred_labels, pred_locs, out_shapes, defaults, batch_index):
    boxes = [
        [[[None for i in range(layer_boxes[o])] for y in range(out_shapes[o][2])] for x in
         range(out_shapes[o][1])]
        for o in range(len(layer_boxes))]
    confidences = []
    index = 0

    for o_i in range(len(layer_boxes)):
        for y in range(out_shapes[o_i][2]):
            for x in range(out_shapes[o_i][1]):
                for i in range(layer_boxes[o_i]):
                    diffs = pred_locs[index]

                    w = defaults[o_i][x][y][i][2] + diffs[2]
                    h = defaults[o_i][x][y][i][3] + diffs[3]

                    cX = defaults[o_i][x][y][i][0] + diffs[0]
                    cY = defaults[o_i][x][y][i][1] + diffs[1]

                    boxes[o_i][x][y][i] = [cX, cY, w, h]
                    logits = pred_labels[index]
                    confidences.append(([o_i, x, y, i], np.amax(np.exp(logits) / (np.sum(np.exp(logits)) + 1e-5)),
                                        np.argmax(logits)))

                    index += 1

    sorted_confidences = sorted(confidences, key=lambda tup: tup[1])[::-1]

    return boxes, sorted_confidences

class Matcher:
    def __init__(self, out_shapes, defaults):
        self.out_shapes = out_shapes
        self.defaults = defaults

    def match_boxes(self, pred_labels, pred_locs, anns, batch_index):
        boxes, sorted_confidences = format_output(pred_labels, pred_locs, self.out_shapes, self.defaults, batch_index)

        matches = [[[[None for i in range(layer_boxes[o])] for y in range(self.out_shapes[o][2])] for x in range(self.out_shapes[o][1])]
                 for o in range(len(layer_boxes))]

        positive_count = 0

        for index, (gtbox, id) in zip(range(len(anns[batch_index])), anns[batch_index]):

            jaccs = []

            for o in range(len(layer_boxes)):
                for y in range(self.out_shapes[o][2]):
                    for x in range(self.out_shapes[o][1]):
                        for i in range(layer_boxes[o]):
                            box = boxes[o][x][y][i]
                            j = calc_jaccard(gtbox, center2cornerbox(box)) #gtbox is corner, box is center
                            jaccs.append(([o, x, y, i], j, (gtbox, id)))

            sorted_jaccs = sorted(jaccs, key=lambda tup: tup[1])[::-1]

            for box, jacc, (gtbox, id) in sorted_jaccs:
                if jacc >= 0.5:
                    matches[box[0]][box[1]][box[2]][box[3]] = (gtbox, id)
                    positive_count += 1

            top_box = sorted_jaccs[0][0]
            if matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] is not None:
                positive_count += 1

            matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] = sorted_jaccs[0][2]

        print("annotations: %i" % len(anns[batch_index]))
        print("positives: %i" % positive_count)
        negative_max = positive_count * 3
        negative_count = 0

        for box, conf, top_label in sorted_confidences:
            if negative_count >= negative_max:
                break

            if matches[box[0]][box[1]][box[2]][box[3]] == None and top_label != classes-1: # if not background class
                matches[box[0]][box[1]][box[2]][box[3]] = -1
                negative_count += 1

        print("negative: %i" % negative_count)

        return boxes, matches