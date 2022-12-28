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


def make_image(img, boxes, labels, img_num, rotation, rect_th=2, text_size=0.5, text_th=1):
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, str(labels[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    # save frame as JPG file
    cv2.imwrite("img_"+str(img_num)+"_"+str(rotation)+".jpg", img)


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
img_num = rot = ''.join(filter(lambda i: i.isdigit(), "img_1053_og.jpg"))
rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
             180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
# for rotation in rotations:
#     # convert to cv2 image
#     image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     h, w = image.shape[:2]
#     cx, cy = (int(w / 2), int(h / 2))
#     # get center
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(
#         image_center, rotation, 1.0)
#     # rotate image
#     result = cv2.warpAffine(
#         image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     # rotate points
#     rotatedBoxes = []
#     for box in boxes:
#         cos, sin = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
#         newW = int((h * sin) + (w * cos))
#         newH = int((h * cos) + (w * sin))
#         rot_mat[0, 2] += (newW / 2) - cx
#         rot_mat[1, 2] += (newH / 2) - cy
#         v = [box[0], box[1], 1]
#         adjusted_coord_min = np.dot(rot_mat, v)
#         v = [box[2], box[3], 1]
#         adjusted_coord_max = np.dot(rot_mat, v)
#         rotatedBoxes.append([int(adjusted_coord_min[0]), int(adjusted_coord_min[1]),
#                             int(adjusted_coord_max[0]), int(adjusted_coord_max[1])])
#         # xmin, ymin = rotate(image_center, (box[0], box[1]), rotation)
#         # xmax, ymax = rotate(image_center, (box[2], box[3]), rotation)
#         # rotatedBoxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
#     # create rotation matrix
#     # save frame as JPG file
#     make_image(result, rotatedBoxes, labels, img_num, rotation)


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle,  cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int 
        height of the image

    w : int 
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  

    Returns 
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
for rotation in rotations:
    image = np.copy(img)
    w, h = image.shape[1], image.shape[0]
    cx, cy = w//2, h//2

    image = rotate_im(image, rotation)
    bboxes = np.array(boxes)
    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:, 4:]))

    corners[:, :8] = rotate_box(corners[:, :8], rotation, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)

    scale_factor_x = image.shape[1] / w

    scale_factor_y = image.shape[0] / h

    image = cv2.resize(image, (w, h))

    new_bbox[:, :4] /= [scale_factor_x,
                        scale_factor_y, scale_factor_x, scale_factor_y]

    bboxes = new_bbox
    print(bboxes)
    # bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)
    make_image(image, bboxes, labels, img_num, rotation)
