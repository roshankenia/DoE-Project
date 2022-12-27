import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from PIL import Image
from xml.dom.minidom import parse
import math

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# make image using box


def make_image(img, boxes, labels, rect_th=2, text_size=2, text_th=2):
    for i in range(len(boxes)):
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]),
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, str(labels[i]), (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    # save frame as JPG file
    cv2.imwrite("test.jpg", img)


# obtain data
img = Image.open("./img_1053_og.jpg").convert("RGB")

# read the annotation files from the path, which are in xml format
dom = parse("./img_1053_x.xml")
# get the element of the annotation files
data = dom.documentElement
# get the objects of the elements
objects = data.getElementsByTagName('object')
# extract the content of the annotation file, which includes class label and bounding box coordinates
boxes = []
labels = []
for object_ in objects:
    # extract the class label, 0 for background, 1 and 2 for mark_type_1 and mark_type_2 respectively
    name = object_.getElementsByTagName(
        'name')[0].childNodes[0].nodeValue
    # labels.append(np.int(name[-1]))
    labels.append(np.int(name.split("_")[-1]))

    # extract the bounding box coordinates
    bndbox = object_.getElementsByTagName('bndbox')[0]
    xmin = np.float64(bndbox.getElementsByTagName(
        'xmin')[0].childNodes[0].nodeValue)
    ymin = np.float64(bndbox.getElementsByTagName(
        'ymin')[0].childNodes[0].nodeValue)
    xmax = np.float64(bndbox.getElementsByTagName(
        'xmax')[0].childNodes[0].nodeValue)
    ymax = np.float64(bndbox.getElementsByTagName(
        'ymax')[0].childNodes[0].nodeValue)
    boxes.append([xmin, ymin, xmax, ymax])

print('boxes:', boxes)
print('labels:', labels)

rotation = 90
# convert to cv2 image
image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# get center
image_center = tuple(np.array(image.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(
    image_center, rotation, 1.0)
# rotate image
result = cv2.warpAffine(
    image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
# rotate points
rotatedBoxes = []
for box in boxes:
    points = [[box[0], box[1], 1], [box[2], box[3], 1]]
    points = np.array(points)
    bb_rotated = np.dot(rot_mat, points).T
    print(bb_rotated)
    # xmin, ymin = rotate(image_center, (box[0], box[1]), rotation)
    # xmax, ymax = rotate(image_center, (box[2], box[3]), rotation)
    # rotatedBoxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
# create rotation matrix
# save frame as JPG file
# make_image(result, rotatedBoxes, labels)
