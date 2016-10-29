import constants as c
from constants import layer_boxes, classes, negposratio
# cant import out_shapes and defaults here since its still not initialized
from ssd_common import center2cornerbox, calc_jaccard
import numpy as np

def format_output(pred_labels, pred_locs, batch_index):

    boxes = [
        [[[None for i in range(layer_boxes[o])] for x in range(c.out_shapes[o][1])] for y in range(c.out_shapes[o][2])]
        for o in range(len(layer_boxes))]
    confidences = []
    index = 0

    for o_i in range(len(layer_boxes)):
        for y in range(c.out_shapes[o_i][2]):
            for x in range(c.out_shapes[o_i][1]):
                for i in range(layer_boxes[o_i]):
                    diffs = pred_locs[index]

                    w = c.defaults[o_i][x][y][i][2] + diffs[2]
                    h = c.defaults[o_i][x][y][i][3] + diffs[3]

                    c_x = c.defaults[o_i][x][y][i][0] + diffs[0]
                    c_y = c.defaults[o_i][x][y][i][1] + diffs[1]

                    boxes[o_i][x][y][i] = [c_x, c_y, w, h]
                    logits = pred_labels[index]
                    # indices, max probability, corresponding label
                    confidences.append(([o_i, x, y, i], np.amax(np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)),
                                        np.argmax(logits)))

                    index += 1

    sorted_confidences = sorted(confidences, key=lambda tup: tup[1])[::-1]

    return boxes, sorted_confidences

class Matcher:
    def __init__(self):
       pass

    def match_boxes(self, pred_labels, pred_locs, anns, batch_index):
        boxes, sorted_confidences = format_output(pred_labels, pred_locs, batch_index)

        matches = [[[[None for i in range(c.layer_boxes[o])] for x in range(c.out_shapes[o][1])] for y in range(c.out_shapes[o][2])]
                 for o in range(len(layer_boxes))]

        positive_count = 0

        for index, (gt_box, id) in zip(range(len(anns[batch_index])), anns[batch_index]):

            jaccs = []

            for o in range(len(layer_boxes)):
                for y in range(c.out_shapes[o][2]):
                    for x in range(c.out_shapes[o][1]):
                        for i in range(layer_boxes[o]):
                            box = boxes[o][x][y][i]
                            j = calc_jaccard(gt_box, center2cornerbox(box)) #gt_box is corner, box is center-based so convert
                            jaccs.append(([o, x, y, i], j, (gt_box, id)))

            sorted_jaccs = sorted(jaccs, key=lambda tup: tup[1])[::-1]

            for box, jacc, (gt_box, id) in sorted_jaccs:
                if jacc >= 0.5:
                    matches[box[0]][box[1]][box[2]][box[3]] = (gt_box, id)
                    positive_count += 1

            top_box = sorted_jaccs[0][0]
            if matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] is not None:
                positive_count += 1

            matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] = sorted_jaccs[0][2]

        negative_max = positive_count * negposratio
        negative_count = 0

        for box, conf, top_label in sorted_confidences:
            if negative_count >= negative_max:
                break

            if matches[box[0]][box[1]][box[2]][box[3]] == None and top_label != classes:  # if not background class
                matches[box[0]][box[1]][box[2]][box[3]] = -1
                negative_count += 1

        #print("negative: %i" % negative_count)

        return boxes, matches