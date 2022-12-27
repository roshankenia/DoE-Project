import math
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
import PIL
from PIL import Image
from PIL import ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# coordinate system conversion methods


def convert_coordinates(geometry, page_dim):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]


def get_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:
                converted_coordinates = convert_coordinates(
                    obj3["geometry"], page_dim
                )
                print("{}: {}".format(converted_coordinates,
                                      obj3["value"]
                                      )
                      )
                text_coordinates.append(converted_coordinates)
    return text_coordinates


# def draw_predictions(img, bounds):


model = ocr_predictor(pretrained=True)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

root = "./ceramicimages"
# need to iterate through each image folder
imageFolders = list(sorted(os.listdir(root)))

for imageFolder in imageFolders:
    if imageFolder == "image301":
        # obtain all image rotations
        rotations = list(sorted(os.listdir(os.path.join(root, imageFolder))))
        rotationsWithRoot = []
        for rotation in rotations:
            print('Rotation:', rotation)
            # rotationsWithRoot.append(os.path.join(root, imageFolder, rotation))
            # # create document image
            # all_rotations_doc = DocumentFile.from_images(rotationsWithRoot)
            # result = model(all_rotations_doc)
            rotation_doc = DocumentFile.from_images(
                os.path.join(root, imageFolder, rotation))

            result = model(rotation_doc)

            output = result.export()
            graphical_coordinates = get_coordinates(output)
