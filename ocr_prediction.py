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

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# need to iterate through each image folder
imageFolders = list(sorted(os.listdir("./ceramicimages")))

print(imageFolders)
