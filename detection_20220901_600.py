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
# from engine import train_one_epoch, evaluate
# utils、transforms、engine are the utils.py、transforms.py、engine.py under this fold

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
        minval = arr[i,:,:].min()
        maxval = arr[i,:,:].max()
        if minval != maxval:
            arr[i,:,:] -= minval
            arr[i,:,:] /= (maxval-minval)
    return arr


# define our own dataset
class PebbleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image and annotation files, sorting them to ensure that the images and their annotations are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        # load images and their corresponding bbox annotations
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
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
        labels = []
        for object_ in objects:
            # extract the class label, 0 for background, 1 and 2 for mark_type_1 and mark_type_2 respectively
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue
            #labels.append(np.int(name[-1]))
            labels.append(np.int(name.split("_")[-1]))

            # extract the bounding box coordinates
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
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


###############################
# Start to train the model
###############################
# change the root path depending on your own dataset path 
root = r'/data/home/stufs1/liahan/DOE/new_datasets_annotations/doe_dataset_GT_white_ink_voc_20220901_600'

# train on the GPU (specify GPU ID with 'cuda:id'), or on the CPU if a GPU is not available
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 11 classes, 0, 1，..., 9, and background
num_classes = 11
# use our dataset with defined transformations
# note that the 'dataset' and 'dataset_test' are with the same images in the same order
# the only difference is that images in 'dataset' can be randomly flipped,
# while images in 'dataset_test' are not flipped
dataset = PebbleDataset(root, get_transform(train=True))
dataset_test = PebbleDataset(root, get_transform(train=False))

# split the dataset (399 images in total) into
# training set (300 images) and test set (99 images)
s_ratio = 300
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:s_ratio])
dataset_test = torch.utils.data.Subset(dataset_test, indices[s_ratio:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, # num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, # num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # or get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# collect the trainable parameters in the model
params = [p for p in model.parameters() if p.requires_grad]

# define the optimizer, here we use SGD
optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

# define a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for a defined number of epochs
num_epochs = 100

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # images and targets are all put '.to(device)' by the 'train_one_epoch' in engine.py
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset    
    evaluate(model, data_loader_test, device=device)    
    
    print('')
    print('==================================================')
    print('')
print("Training is done!")


# save the trained model
torch.save(model, r'./saved_model/model_doe_20220901_600_multi_class_100epochs.pkl')


def showbbox(model, img, idx):
    # the input images are tensors with values in [0, 1]
    #print("input image shape...:", type(img))
    image_array = img.numpy()
    image_array = np.array(normalize(image_array),dtype=np.float32) 
    img = torch.from_numpy(image_array)
    
    model.eval()
    with torch.no_grad():
        '''
        prediction is in the following format：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        
        prediction = model([img.to(device)])
        
    print(prediction)
        
    img = img.permute(1,2,0)  # C,H,W → H,W,C
    img = (img * 255).byte().data.cpu()  # [0, 1] → [0, 255]
    img = np.array(img)  # tensor → ndarray
       
    # for i in range(prediction[0]['boxes'].cpu().shape[0]): # select all the predicted bounding boxes
    for i in range(3): # select the top-3 predicted bounding boxes
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        
        label = prediction[0]['labels'][i].item()
        
        if label == 10: # start with background as 0
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=1)
            cv2.putText(img, '0', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=2)
        elif label == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)
            cv2.putText(img, '1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)   
        elif label == 2:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
            cv2.putText(img, '2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
        elif label == 3:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 100, 255), thickness=1)
            cv2.putText(img, '3', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), thickness=2)
        elif label == 4:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 100, 100), thickness=1)
            cv2.putText(img, '4', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), thickness=2)
        elif label == 5:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 255), thickness=1)
            cv2.putText(img, '5', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=2)
        elif label == 6:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), thickness=1)
            cv2.putText(img, '6', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=2)
        elif label == 7:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness=1)
            cv2.putText(img, '7', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), thickness=2)
        elif label == 8:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (100, 0, 0), thickness=1)
            cv2.putText(img, '8', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 0), thickness=2)
        elif label == 9:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 100), thickness=1)
            cv2.putText(img, '9', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), thickness=2)            
    
    # plot the output images with prediction result
#    plt.figure(figsize=(50,50))
#    plt.imshow(img)
#    plt.axis('off')
    vis_tgt_path = "./visualization_results/"
    if not os.path.isdir(vis_tgt_path):
        os.mkdir(vis_tgt_path)
    plt.savefig(os.path.join(vis_tgt_path, "sample_" + f"{idx:02d}" + "_vis.png"))


# check the result
model = torch.load(r'./saved_model/model_doe_20220901_600_multi_class_100epochs.pkl')
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for idx in range(len(dataset_test)):
    img, _ = dataset_test[idx] 
    showbbox(model, img, idx)


