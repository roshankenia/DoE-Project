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
    #print("...arr shape", arr.shape)
    #print("arr shape: ", arr.shape)
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
                    0.5, (255, 0, 0), thickness=2)
    elif label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=1)
        cv2.putText(img, '1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), thickness=2)
    elif label == 2:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 255), thickness=1)
        cv2.putText(img, '2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), thickness=2)
    elif label == 3:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 100, 255), thickness=1)
        cv2.putText(img, '3', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 100, 255), thickness=2)
    elif label == 4:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 100, 100), thickness=1)
        cv2.putText(img, '4', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 100, 100), thickness=2)
    elif label == 5:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 0, 255), thickness=1)
        cv2.putText(img, '5', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), thickness=2)
    elif label == 6:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 255), thickness=1)
        cv2.putText(img, '6', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), thickness=2)
    elif label == 7:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 255, 0), thickness=1)
        cv2.putText(img, '7', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), thickness=2)
    elif label == 8:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (100, 0, 0), thickness=1)
        cv2.putText(img, '8', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 0, 0), thickness=2)
    elif label == 9:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 100), thickness=1)
        cv2.putText(img, '9', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 100), thickness=2)


def showbbox(img, bbox, id):
    # the input images are tensors with values in [0, 1]
    #print("input image shape...:", type(img))
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
    vis_tgt_path = "./visualization_results/SVHNdata/"
    if not os.path.isdir(vis_tgt_path):
        os.mkdir(vis_tgt_path)
    cv2.imwrite(os.path.join(vis_tgt_path, "sample_" +
                str(id) + "_prediction.jpg"), img)


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
for i in range(len(names)):
    imageName = names[i]
    bbox = bboxes[i]

    imgNum = ''.join(filter(lambda i: i.isdigit(), imageName))

    img = Image.open(os.path.join(imageroot, imageName)).convert("RGB")

    img, _ = transform(img, None)
    showbbox(img, bbox, imgNum)
    break
