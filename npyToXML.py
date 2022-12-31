import sys
import transforms as T
import utils
from engine import train_one_epoch, evaluate
from xml.dom.minidom import parse
from PIL import Image
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import cv2
import os
import torch
import numpy as np
import xml
from xml.dom.minidom import parse
import math
import shutil
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, dump

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')
# utils transforms, engine are the utils.py, transforms.py, engine.py under this fold

# %matplotlib inline


def normalize(arr):
    """
    Linear normalization
    normalize the input array value into [0, 1]
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # print("...arr shape", arr.shape)
    # print("arr shape: ", arr.shape)
    for i in range(3):
        minval = arr[i, :, :].min()
        maxval = arr[i, :, :].max()
        if minval != maxval:
            arr[i, :, :] -= minval
            arr[i, :, :] /= (maxval-minval)
    return arr


def fig_draw(img, bbox, label):
    # draw predicted bounding box and class label on the input image
    xmin = round(bbox[0])
    ymin = round(bbox[1])
    xmax = round(bbox[2])
    ymax = round(bbox[3])

    label = int(label)

    if label == 10:  # start with background as 0
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 0, 0), thickness=1)
        cv2.putText(img, '0', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), thickness=1)
    elif label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=1)
        cv2.putText(img, '1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), thickness=1)
    elif label == 2:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 255), thickness=1)
        cv2.putText(img, '2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), thickness=1)
    elif label == 3:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 100, 255), thickness=1)
        cv2.putText(img, '3', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 100, 255), thickness=1)
    elif label == 4:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 100, 100), thickness=1)
        cv2.putText(img, '4', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 100, 100), thickness=1)
    elif label == 5:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 0, 255), thickness=1)
        cv2.putText(img, '5', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), thickness=1)
    elif label == 6:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 255), thickness=1)
        cv2.putText(img, '6', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), thickness=1)
    elif label == 7:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 255, 0), thickness=1)
        cv2.putText(img, '7', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), thickness=1)
    elif label == 8:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (100, 0, 0), thickness=1)
        cv2.putText(img, '8', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 0, 0), thickness=1)
    elif label == 9:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 100), thickness=1)
        cv2.putText(img, '9', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 100), thickness=1)


def showbbox(img, bbox, id, annotationVisPath):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
    img = np.array(img)  # tensor -> ndarray

    for j in range(len(bbox)):
        digitBox = bbox[j][0:4]
        label = bbox[j][4]
        fig_draw(img, digitBox, label)
    # save frame as JPG file
    cv2.imwrite(os.path.join(annotationVisPath,
                "img_" + str(id) + ".jpg"), img)


def makeXML(imgNum, annotationPath, bboxs):
    # create copy xml

    shutil.copy2("./example.xml", os.path.join(annotationPath,
                 "img_" + str(imgNum) + ".xml"))

    tree = ElementTree()
    tree.parse(os.path.join(annotationPath, "img_" + str(imgNum) + ".xml"))

    filename = tree.find('filename')
    filename.text = "img_" + str(imgNum) + ".png"

    annotation = tree.getroot()

    for bbox in bboxs:
        # make bbox object
        objectElement = ET.Element("object")
        nameElement = ET.Element("name")
        label = int(bbox[4])
        if label == 10:
            # convert 10 to 0
            label = 0
        nameElement.text = str(label)
        objectElement.append(nameElement)
        bndBoxElement = ET.Element("bndbox")
        xminElement = ET.Element("xmin")
        xminElement.text = str(bbox[0])
        bndBoxElement.append(xminElement)
        yminElement = ET.Element("ymin")
        yminElement.text = str(bbox[1])
        bndBoxElement.append(yminElement)
        xmaxElement = ET.Element("xmax")
        xmaxElement.text = str(bbox[2])
        bndBoxElement.append(xmaxElement)
        ymaxElement = ET.Element("ymax")
        ymaxElement.text = str(bbox[3])
        bndBoxElement.append(ymaxElement)
        objectElement.append(bndBoxElement)
        # append to annotation
        annotation.append(objectElement)

    tree.write(os.path.join(annotationPath, "img_" + str(imgNum) + ".xml"))


# check the result
# train on the GPU (specify GPU ID with 'cuda:id'), or on the CPU if a GPU is not available
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 11 classes, 0, 1, ..., 9, and background
num_classes = 11
# need to iterate through each image folder
transform = T.Compose([T.PILToTensor()])
names = np.load('SVHNname.npy')
bboxes = np.load('SVHNbbox.npy', allow_pickle=True)

imageroot = '../SVHN/test/'

# make directories
root = "./SVHNData/"
if not os.path.isdir(root):
    os.mkdir(root)
annotationPath = "./SVHNData/Annotations/"
if not os.path.isdir(annotationPath):
    os.mkdir(annotationPath)
annotationVisPath = "./SVHNData/AnnotationsVisualization/"
if not os.path.isdir(annotationVisPath):
    os.mkdir(annotationVisPath)
# x = 0
for i in range(len(names)):
    imageName = names[i]
    bbox = bboxes[i]

    imgNum = ''.join(filter(lambda i: i.isdigit(), imageName))
    # print("IMG:", imgNum)

    img = Image.open(os.path.join(imageroot, imageName)).convert("RGB")
    width, height = img.size
    if width > 500 or height > 500:
        print(width, height)
    img, _ = transform(img, None)
    # showbbox(img, bbox, imgNum, annotationVisPath)
    makeXML(imgNum, annotationPath, bbox)
    # x += 1
    # if x == 10:
    #     break
