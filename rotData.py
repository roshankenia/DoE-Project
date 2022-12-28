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
import shutil
from xml.etree.ElementTree import ElementTree, dump


# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


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

# make image using box


def make_image(img, boxes, labels, img_num, rotation, annotationVisPath, rect_th=2, text_size=0.5, text_th=1):
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, str(labels[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    # save frame as JPG file
    cv2.imwrite(os.path.join(annotationVisPath, "img_" +
                str(img_num) + "_"+str(rotation)+"_annotated.jpg"), img)


# make directories
root = "./RotatedData/"
if not os.path.isdir(root):
    os.mkdir(root)
annotationPath = "./RotatedData/Annotations/"
if not os.path.isdir(annotationPath):
    os.mkdir(annotationPath)
annotationVisPath = "./RotatedData/AnnotationsVisualization/"
if not os.path.isdir(annotationVisPath):
    os.mkdir(annotationVisPath)
jpegpath = "./RotatedData/JPEGImages/"
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)

sourceRoot = "./new_datasets_annotations/doe_dataset_GT_ceramic_painted_voc_20220901_600/"
sourceJPEGPath = os.path.join(sourceRoot, "JPEGImages")
sourceAnnotationPath = os.path.join(sourceRoot, "Annotations")

# obtain all JPEG image file names
allJPEGImgs = list(sorted(os.listdir(sourceJPEGPath)))
x = 0
for imageFile in allJPEGImgs:
    # obtain data
    imagePIL = Image.open(os.path.join(
        sourceJPEGPath, imageFile)).convert("RGB")
    img_num = ''.join(filter(lambda i: i.isdigit(), imageFile))

    # read the annotation files from the path, which are in xml format
    dom = parse(os.path.join(sourceAnnotationPath, "img_"+str(img_num)+".xml"))
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
    rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
                 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for rotation in rotations:
        print('\nRotation:', rotation)
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
        # save frame as JPG file

        cv2.imwrite(os.path.join(jpegpath, "img_" +
                    str(img_num) + "_"+str(rotation)+".jpg"), image)
        # bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)
        make_image(image, bboxes, labels, img_num, rotation)

        # create copy xml

        shutil.copy2(os.path.join(sourceAnnotationPath, "img_"+str(img_num) + ".xml"),
                     os.path.join(annotationPath, "img_" + str(img_num) + "_"+str(rotation)+".xml"))

        tree = ElementTree()
        tree.parse(os.path.join(annotationPath, "img_" +
                   str(img_num) + "_"+str(rotation)+".xml"))

        filename = tree.find('filename')
        filename.text = "img_" + str(img_num) + "_"+str(rotation)+".jpg"

        objects = tree.findall('object')
        for o in range(len(objects)):
            object = objects[o]
            bbox = bboxes[o]
            bndbox = object.find('bndbox')
            xmin, ymin, xmax, ymax = list(bndbox)
            print('Old:', xmin.text, ymin.text, xmax.text, ymax.text)
            print('New:', bbox[0], bbox[1], bbox[2], bbox[3])
            xmin.text = str(bbox[0])
            ymin.text = str(bbox[1])
            xmax.text = str(bbox[2])
            ymax.text = str(bbox[3])

        tree.write(os.path.join(annotationPath, "img_" +
                   str(img_num) + "_"+str(rotation)+".xml"))
        x += 1
        if x == 5:
            break
