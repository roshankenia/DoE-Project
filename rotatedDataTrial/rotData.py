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

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


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
