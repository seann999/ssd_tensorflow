import cv2

image_size = 500.0
layer_boxes = [3, 6, 6, 6, 6, 6] 
classes = 81
batch_size = 2
out_shapes = None

indices2index = None

def draw_rect(I, r, c, thickness=1):
    if abs(sum(r)) < 100:
        cv2.rectangle(I, (int(r[0] * image_size), int(r[1] * image_size)),
                      (int((r[0] + max(r[2], 0)) * image_size), int((r[1] + max(r[3], 0)) * image_size)),
                      c, thickness)

def center2cornerbox(rect):
    return [rect[0] - rect[2]/2.0, rect[1] - rect[3]/2.0, rect[2], rect[3]]

def corner2centerbox(rect):
    return [rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0, rect[2], rect[3]]
