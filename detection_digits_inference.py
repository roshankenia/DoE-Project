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

# train on the GPU (specify GPU ID with 'cuda:id'), or on the CPU if a GPU is not available
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')


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


def fig_draw(img, prediction, idx):
    # draw predicted bounding box and class label on the input image
    i = idx
    xmin = round(prediction[0]['boxes'][i][0].item())
    ymin = round(prediction[0]['boxes'][i][1].item())
    xmax = round(prediction[0]['boxes'][i][2].item())
    ymax = round(prediction[0]['boxes'][i][3].item())

    label = prediction[0]['labels'][i].item()

    if label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=1)
        cv2.putText(img, 'digits', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), thickness=2)


def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        xmin = round(bbox[0])
        ymin = round(bbox[1])
        xmax = round(bbox[2])
        ymax = round(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=1)
        cv2.putText(img, 'digits', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), thickness=2)


def create_overlap_box(bbox1, bbox2):
    # check if these two bboxes overlap
    xmin1 = bbox1[0]
    ymin1 = bbox1[1]
    xmax1 = bbox1[2]
    ymax1 = bbox1[3]
    # obtain corner points
    upperLeftX1 = xmin1
    upperRightX1 = xmax1
    lowerLeftX1 = upperLeftX1
    lowerRightX1 = upperRightX1
    upperLeftY1 = ymin1
    upperRightY1 = upperLeftY1
    lowerLeftY1 = ymax1
    lowerRightY1 = lowerLeftY1

    xmin2 = bbox2[0]
    ymin2 = bbox2[1]
    xmax2 = bbox2[2]
    ymax2 = bbox2[3]
    # obtain corner points
    upperLeftX2 = xmin2
    upperRightX2 = xmax2
    lowerLeftX2 = upperLeftX2
    lowerRightX2 = upperRightX2
    upperLeftY2 = ymin2
    upperRightY2 = upperLeftY2
    lowerLeftY2 = ymax2
    lowerRightY2 = lowerLeftY2

    overlaps = False
    # CHECK if any point overlaps with another
    if lowerLeftX1 < upperLeftX2 < upperRightX1 and lowerLeftY1 < upperLeftY2 < upperRightY1:
        overlaps = True
    if lowerLeftX1 < upperRightX2 < upperRightX1 and lowerLeftY1 < upperRightY2 < upperRightY1:
        overlaps = True
    if lowerLeftX1 < lowerRightX2 < upperRightX1 and lowerLeftY1 < lowerRightY2 < upperRightY1:
        overlaps = True
    if lowerLeftX1 < lowerLeftX2 < upperRightX1 and lowerLeftY1 < lowerLeftY2 < upperRightY1:
        overlaps = True
    if overlaps:
        # create new bounding box
        newXMin = min(xmin1, xmin2)
        newYMin = min(ymin1, ymin2)
        newXMax = max(xmax1, xmax2)
        newYMax = max(ymax1, ymax2)
        return [newXMin, newYMin, newXMax, newYMax]
    return None


def create_new_bboxes(bboxes):
    newBBoxes = []
    # check first 2
    newBB1 = create_overlap_box(bboxes[0], bboxes[1])
    if newBB1 != None:
        newBB2 = create_overlap_box(newBB1, bboxes[2])
        if newBB2 != None:
            newBBoxes.append(newBB2)
            return newBBoxes
        else:
            newBBoxes.append(newBB1)
            newBBoxes.append(bboxes[2])
            return newBBoxes
    # check other two combinations
    newBB1 = create_overlap_box(bboxes[0], bboxes[2])
    if newBB1 != None:
        newBB2 = create_overlap_box(newBB1, bboxes[1])
        if newBB2 != None:
            newBBoxes.append(newBB2)
            return newBBoxes
        else:
            newBBoxes.append(newBB1)
            newBBoxes.append(bboxes[1])
            return newBBoxes
    newBB1 = create_overlap_box(bboxes[1], bboxes[2])
    if newBB1 != None:
        newBB2 = create_overlap_box(newBB1, bboxes[0])
        if newBB2 != None:
            newBBoxes.append(newBB2)
            return newBBoxes
        else:
            newBBoxes.append(newBB1)
            newBBoxes.append(bboxes[0])
            return newBBoxes

    # otherwise return top 3
    return [bboxes[0], bboxes[1], bboxes[2]]


def showbbox(model, img, idx):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
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

    print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
    img = np.array(img)  # tensor -> ndarray

    bboxes = prediction[0]['boxes'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    goodBBoxes = []
    for i in range(len(scores)):
        if scores[i] >= 0.4:
            goodBBoxes.append(bboxes[i])
        else:
            # scores already sorted
            break

    # for i in range(prediction[0]['boxes'].cpu().shape[0]): # select all the predicted bounding boxes
    if len(goodBBoxes) >= 3:
        # combine those that overlap
        newBBoxes = create_new_bboxes(goodBBoxes)
        draw_bboxes(img, newBBoxes)
    elif len(goodBBoxes) == 2:
        # check if two overlap
        combined = create_overlap_box(goodBBoxes[0], goodBBoxes[1])
        if combined != None:
            draw_bboxes(img, [combined])
        else:
            draw_bboxes(img, goodBBoxes)
    else:
        draw_bboxes(img, goodBBoxes)

    vis_tgt_path = "./cropCerDigitDetection/"
    if not os.path.isdir(vis_tgt_path):
        os.mkdir(vis_tgt_path)
    cv2.imwrite(os.path.join(vis_tgt_path, str(idx) + "_vis.jpg"), img)


# check the result
model = torch.load(
    r'./saved_model/digits_detector.pkl')
model.to(device)
transform = T.Compose([T.PILToTensor()])
imgNames = list(sorted(os.listdir("./bestceramiccrop")))
for imgName in imgNames:
    image = Image.open(os.path.join(
        "./bestceramiccrop/", imgName)).convert("RGB")
    img, _ = transform(image, None)
    num = ''.join(filter(lambda i: i.isdigit(), imgName))
    showbbox(model, img, num)
