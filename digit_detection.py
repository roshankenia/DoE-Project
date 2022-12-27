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
# train on the GPU (specify GPU ID with 'cuda:id'), or on the CPU if a GPU is not available
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 11 classes, 0, 1, ..., 9, and background
num_classes = 11
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


def fig_draw(img, prediction, idx):
    # draw predicted bounding box and class label on the input image
    i = idx
    xmin = round(prediction[0]['boxes'][i][0].item())
    ymin = round(prediction[0]['boxes'][i][1].item())
    xmax = round(prediction[0]['boxes'][i][2].item())
    ymax = round(prediction[0]['boxes'][i][3].item())

    label = prediction[0]['labels'][i].item()

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


def showbbox(model, img, rot, output_path):
    # the input images are tensors with values in [0, 1]
    #print("input image shape...:", type(img))
    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    model.eval()
    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = model([img.to(device)])

    # print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
    img = np.array(img)  # tensor -> ndarray

    # for i in range(prediction[0]['boxes'].cpu().shape[0]): # select all the predicted bounding boxes
    if len(prediction[0]['labels']) >= 3:
        for i in range(3):  # select the top-3 predicted bounding boxes
            fig_draw(img, prediction, i)
    else:
        for i in range(len(prediction[0]['labels'])):
            fig_draw(img, prediction, i)
    # save frame as JPG file
    cv2.imwrite(os.path.join(output_path, str(rot)+"_prediction.jpg"), img)


# check the result
model = torch.load(
    r'./saved_model/model_doe_ceramic_paint_fastRCNN_v2_200epoch.pkl')
model.to(device)

root = "./ceramicimages"
# need to iterate through each image folder
imageFolders = list(sorted(os.listdir(root)))
transform = T.Compose([T.PILToTensor()])

for imageFolder in imageFolders:
    if imageFolder == "image300":
        # obtain all image rotations
        rotations = list(sorted(os.listdir(os.path.join(root, imageFolder))))
        rotationsWithRoot = []
        for rotation in rotations:
            rot = ''.join(filter(lambda i: i.isdigit(), rotation))
            print('Rotation:', rot)
            # rotationsWithRoot.append(os.path.join(root, imageFolder, rotation))
            # # create document image
            # all_rotations_doc = DocumentFile.from_images(rotationsWithRoot)
            # result = model(all_rotations_doc)
            img = Image.open(os.path.join(
                root, imageFolder, rotation)).convert("RGB")

            img = transform(img)

            output_path = os.path.join(root, imageFolder)

            showbbox(model, img, rot, output_path)
