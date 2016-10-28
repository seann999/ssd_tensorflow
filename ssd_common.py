import cv2
from constants import image_size

def draw_rect(I, r, c, thickness=1):
    if abs(sum(r)) < 100: # conditional to prevent min/max error
        cv2.rectangle(I, (int(r[0] * image_size), int(r[1] * image_size)),
                      (int((r[0] + max(r[2], 0)) * image_size), int((r[1] + max(r[3], 0)) * image_size)),
                      c, thickness)

def draw_ann(I, r, text):
    draw_rect(I, r, (255, 0, 255), 2)
    cv2.rectangle(I, (int(r[0] * image_size), int(r[1] * image_size - 15)),
                  (int(r[0] * image_size + 100), int(r[1] * image_size)),
                  (255, 0, 255), -1)
    cv2.putText(I, text, (int(r[0] * image_size), int((r[1]) * image_size)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def center2cornerbox(rect):
    return [rect[0] - rect[2]/2.0, rect[1] - rect[3]/2.0, rect[2], rect[3]]

def corner2centerbox(rect):
    return [rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0, rect[2], rect[3]]

def calc_intersection(r1, r2):
    left = max(r1[0], r2[0])
    right = min(r1[0] + r1[2], r2[0] + r2[2])
    bottom = min(r1[1] + r1[3], r2[1] + r2[3])
    top = max(r1[1], r2[1])

    if left < right and top < bottom:
        return (right - left) * (bottom - top)

    return 0


def calc_jaccard(r1, r2):
    r1_ = [r1[0], r1[1], max(r1[2], 0.01), max(r1[3], 0.01)]
    r2_ = [r2[0], r2[1], max(r2[2], 0.01), max(r2[3], 0.01)]
    intersection = calc_intersection(r1_, r2_)
    union = r1_[2] * r1_[3] + r2_[2] * r2_[3] - intersection

    if union <= 0:
        return 0

    j = intersection / union

    return j