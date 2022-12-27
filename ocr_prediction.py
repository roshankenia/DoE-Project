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


model = ocr_predictor(pretrained=True)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

root = "./ceramicimages"
# need to iterate through each image folder
imageFolders = list(sorted(os.listdir(root)))

for imageFolder in imageFolders:
    # obtain all image rotations
    rotations = list(sorted(os.listdir(os.path.join(root, imageFolder))))
    rotationsWithRoot = []
    for rotation in rotations:
        rotationsWithRoot.append(os.path.join(root, imageFolder, rotation))
    # create document image
    all_rotations_doc = DocumentFile.from_images(rotationsWithRoot)

    result = model(all_rotations_doc)
