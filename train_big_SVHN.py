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
class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image and annotation files, sorting them to ensure that the images and their annotations are aligned
        self.xmls = list(
            sorted(os.listdir(os.path.join(self.root, "Annotations"))))

    def __getitem__(self, idx):
        # load images and their corresponding bbox annotations
        # obtain XML first
        xml_path = os.path.join(self.root, "Annotations", self.xmls[idx])
        # read the annotation files from the path, which are in xml format
        dom = parse(xml_path)
        # get the element of the annotation files
        data = dom.documentElement

        # get image name
        fileName = data.getElementsByTagName(
            'filename')[0].childNodes[0].nodeValue
        imgNum = ''.join(filter(lambda i: i.isdigit(), fileName))
        imgPath = os.path.join(self.root, "PNGImages", str(imgNum)+".png")
        img = Image.open(imgPath).convert("RGB")
        # normalization
        # imaaag = np.array(img)
        # print("IMG SIZE:", imaaag.size)
        #img = normalize(img)
        #img = Image.fromarray(normalize(img),'RGB')
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
            xmin = np.float(bndbox.getElementsByTagName(
                'xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName(
                'ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName(
                'xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName(
                'ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

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
        return len(self.xmls)


def get_transform(train):
    transf = []
    # convert the image, a PIL image, into a PyTorch Tensor
    transf.append(T.PILToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth with 50% probability for data augmentation
        transf.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transf)


###############################
# Start to train the model
###############################
# change the root path depending on your own dataset path
root = "./SVHNBigData/"

# train on the GPU (specify GPU ID with 'cuda:id'), or on the CPU if a GPU is not available
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 10 classes, 0, 1, ..., 9
num_classes = 10
# use our dataset with defined transformations
# note that the 'dataset' and 'dataset_test' are with the same images in the same order
# the only difference is that images in 'dataset' can be randomly flipped,
# while images in 'dataset_test' are not flipped
dataset = SVHNDataset(root, get_transform(train=True))
dataset_test = SVHNDataset(root, get_transform(train=False))


# split the dataset (399 images in total) into
# training set (300 images) and test set (99 images)
# s_ratio = 2500
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:s_ratio])
# dataset_test = torch.utils.data.Subset(
#     dataset_test, indices[s_ratio:s_ratio+500])
print("number of images in the training set:", len(dataset))
print("number of images in the testing set:", len(dataset_test))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True,  # num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=16, shuffle=False,  # num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = models.detection.fasterrcnn_resnet50_fpn_v2(
    pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # or get_object_detection_model(num_classes)

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
num_epochs = 100

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # images and targets are all put '.to(device)' by the 'train_one_epoch' in engine.py
    train_one_epoch(model, optimizer, data_loader,
                    device, epoch, print_freq=50)
    # update the learning rate
    lr_scheduler.step()

    if epoch % 25 == 0:
        evaluate(model, data_loader_test, device=device)

    print('')
    print('==================================================')
    print('')
print("Training is done!")

# evaluate on the test dataset
evaluate(model, data_loader_test, device=device)
# save the trained model
torch.save(model, r'./saved_model/SVHN_model_big_100epoch.pkl')
