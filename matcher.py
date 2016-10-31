import constants as c
from constants import layer_boxes, classes, negposratio
# cant import out_shapes and defaults here since its still not initialized
from ssd_common import center2cornerbox, calc_jaccard
import numpy as np

def format_output(pred_labels, pred_locs):

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
                    #if np.argmax(logits) != classes+1:
                        # indices, max probability, corresponding label
                    confidences.append(([o_i, x, y, i], np.amax(np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)), np.argmax(logits)))
                    #else:
                    #    logits = pred_labels[index][:-1]
                    #    confidences.append(([o_i, x, y, i], np.amax(np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)),
                    #                        np.argmax(logits)))

                    index += 1

    sorted_confidences = sorted(confidences, key=lambda tup: tup[1])[::-1]

    return boxes, sorted_confidences

def sort_confidences(pred_labels):
    confidences = []
    index = 0

    for o_i in range(len(layer_boxes)):
        for y in range(c.out_shapes[o_i][2]):
            for x in range(c.out_shapes[o_i][1]):
                for i in range(layer_boxes[o_i]):
                    logits = pred_labels[index]
                    probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)
                    confidences.append(([o_i, x, y, i], np.amax(probs),
                                        np.argmax(logits)))

                    index += 1

    sorted_confidences = sorted(confidences, key=lambda tup: tup[1])[::-1]

    return sorted_confidences

class Matcher:
    def __init__(self):
       pass

    def match_boxes(self, pred_labels, anns):
        sorted_confidences = sort_confidences(pred_labels)

        matches = [[[[None for i in range(c.layer_boxes[o])] for x in range(c.out_shapes[o][1])] for y in range(c.out_shapes[o][2])]
                 for o in range(len(layer_boxes))]

        positive_count = 0

        for index, (gt_box, id) in zip(range(len(anns)), anns):

            top_match = (None, 0)

            for o in range(len(layer_boxes)):
                x1 = max(int(gt_box[0] / (1.0 / c.out_shapes[o][2])), 0)
                y1 = max(int(gt_box[1] / (1.0 / c.out_shapes[o][1])), 0)
                x2 = min(int((gt_box[0] + gt_box[2]) / (1.0 / c.out_shapes[o][2]))+2, c.out_shapes[o][2])
                y2 = min(int((gt_box[1] + gt_box[3]) / (1.0 / c.out_shapes[o][1]))+2, c.out_shapes[o][1])

                for y in range(y1, y2):
                    for x in range(x1, x2):
                        for i in range(layer_boxes[o]):
                            box = c.defaults[o][x][y][i]
                            jacc = calc_jaccard(gt_box, center2cornerbox(box)) #gt_box is corner, box is center-based so convert
                            if jacc >= 0.5:
                                matches[o][x][y][i] = (gt_box, id)
                                positive_count += 1
                            if jacc > top_match[1]:
                                top_match = ([o, x, y, i], jacc)

            top_box = top_match[0]
            #if box's jaccard is <0.5 but is the best
            if top_box is not None and matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] is None:
                positive_count += 1
                matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] = (gt_box, id)

        negative_max = positive_count * negposratio
        negative_count = 0

        for box, conf, top_label in sorted_confidences:
            if negative_count >= negative_max:
                break

            if matches[box[0]][box[1]][box[2]][box[3]] is None and top_label != classes:
                matches[box[0]][box[1]][box[2]][box[3]] = -1
                negative_count += 1

        #print("%i positives" % positive_count)
        #print("%i/%i negatives" % (negative_count, negative_max))

        return matches
