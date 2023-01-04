import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
from xml.dom.minidom import parse
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
# ensure we are running on the correct gpu
import sys
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


# define our own dataset
class DigitMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image and annotation files, sorting them to ensure that the images and their annotations are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(
            sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, idx):
        # load images and their corresponding bbox annotations
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(
            self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")
        # normalization
        #img = np.array(img)
        #img = normalize(img)
        #img = Image.fromarray(normalize(img),'RGB')

        # read the annotation files from the path, which are in xml format
        dom = parse(bbox_xml_path)
        # get the element of the annotation files
        data = dom.documentElement
        # get the objects of the elements
        objects = data.getElementsByTagName('object')
        # extract the content of the annotation file, which includes class label and bounding box coordinates
        boxes = []
        num_objs = 0
        for object_ in objects:
            # extract the bounding box coordinates
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName(
                'xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName(
                'ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName(
                'xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName(
                'ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])
            num_objs += 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        # calculate the area of the bounding box of each instance
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # https://github.com/pytorch/vision/tree/master/references/detection
            # On this website, there are transform examples of RandomHorizontalFlip for target in transforms.py
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transf = []
    # convert the image, a PIL image, into a PyTorch Tensor
    transf.append(T.PILToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth with 50% probability for data augmentation
        transf.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transf)


def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


###############################
# Start to train the model
###############################
# change the root path depending on your own dataset path
root = r'./DigitMaskDataset/'

# train on the GPU (specify GPU ID with 'cuda:id'), or on the CPU if a GPU is not available
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 2 classes digits or background
num_classes = 2
# use our dataset with defined transformations
# note that the 'dataset' and 'dataset_test' are with the same images in the same order
# the only difference is that images in 'dataset' can be randomly flipped,
# while images in 'dataset_test' are not flipped
dataset = DigitMaskDataset(root, get_transform(train=True))
dataset_test = DigitMaskDataset(root, get_transform(train=False))
print("number of images in the training set:", len(dataset))
print("number of images in the testing set:", len(dataset_test))

"""
# split the dataset (399 images in total) into
# training set (300 images) and test set (99 images)
s_ratio = 300
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:s_ratio])
dataset_test = torch.utils.data.Subset(dataset_test, indices[s_ratio:])
"""

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True,  # num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False,  # num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
# model = models.detection.fasterrcnn_resnet50_fpn_v2(
#     pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # or get_object_detection_model(num_classes)
model = build_model(num_classes)

# move model to the right device
model.to(device)

# collect the trainable parameters in the model
params = [p for p in model.parameters() if p.requires_grad]

# define the optimizer, here we use SGD
optimizer = torch.optim.SGD(
    params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

# define a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2)

# let's train it for a defined number of epochs
num_epochs = 50

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # images and targets are all put '.to(device)' by the 'train_one_epoch' in engine.py
    train_one_epoch(model, optimizer, data_loader,
                    device, epoch, print_freq=5)
    # update the learning rate
    lr_scheduler.step()

    if epoch+1 % 10 == 0:
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print('')
    print('==================================================')
    print('')
print("Training is done!")


# save the trained model
torch.save(model, r'./saved_model/digits_detector.pkl')
