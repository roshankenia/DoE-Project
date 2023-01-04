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

    if label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=1)
        cv2.putText(img, 'digits', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), thickness=2)


def showbbox(model, img, idx):
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

    print(prediction)

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

    plt.figure(figsize=(50, 50))
    plt.imshow(img)
    plt.axis('off')
    vis_tgt_path = "./visualization_results/cropCerDigitDetection/"
    if not os.path.isdir(vis_tgt_path):
        os.mkdir(vis_tgt_path)
    plt.savefig(os.path.join(vis_tgt_path, str(idx) + "_vis.jpg"))
    plt.close()


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
