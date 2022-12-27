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

# set to evaluation mode
model = torch.load('mask-rcnn-pebble.pt')
model.eval()
CLASS_NAMES = ['__background__', 'pebble']
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [
        255, 0, 255], [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    if len(pred_t) == 0:
        return None, None, None
    pred_t = pred_t[-1]
    masks = (pred[0]['masks'] > 0.5).detach().cpu().numpy()
    masks = masks.reshape(-1, *masks.shape[-2:])
    print(masks.shape)
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i]
                  for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def make_mask_image(img, masks, boxes, pred_cls, ind, rect_th=2, text_size=2, text_th=2):
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    # save frame as JPG file
    cv2.imwrite("./ceramicimages/image"+str(ind) + "_mask.jpg", img)


def crop_pebble(img, masks, boxes, ind):
    mask = np.asarray(masks[0], dtype="uint8")
    # obtain only the mask pixels from the image
    only_mask = cv2.bitwise_and(img, img, mask=mask)
    bbox = boxes[0]
    # crop the image to only contain the pebble
    crop = only_mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # put pebble on standard 1000x1000 image
    background = np.zeros((1000, 1000, 3), np.uint8)
    ch, cw = crop.shape[:2]

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((1000-ch)/2)
    xoff = round((1000-cw)/2)

    background[yoff:yoff+ch, xoff:xoff+cw] = crop
    # save crop as JPG file
    cv2.imwrite("./ceramicimages/image"+str(ind) + "_crop.jpg", background)

    return background


vidcap = cv2.VideoCapture('Moving Pebbles - Ceramic Paint.MOV')
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('video has', str(frame_count), 'frames.')

rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
             165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315]
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

count = 0
while (vidcap.isOpened()):
    hasFrames, image = vidcap.read()
    if hasFrames:
        if count == 1950 or count == 1952:
            # save unmodified image
            # save frame as JPG file
            cv2.imwrite("./ceramicimages/image" +
                        str(count) + "_unmodified.jpg", image)
            # check if image has a pebble
            masks, boxes, pred_cls = get_prediction(image, .9)
            if masks is not None:
                if len(masks) == 1:
                    make_mask_image(np.copy(image), masks,
                                    boxes, pred_cls, count)
                    image = crop_pebble(np.copy(image), masks, boxes, count)
                    for rotation in rotations:
                        image_center = tuple(np.array(image.shape[1::-1]) / 2)
                        rot_mat = cv2.getRotationMatrix2D(
                            image_center, rotation, 1.0)
                        result = cv2.warpAffine(
                            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                        # sharpened = cv2.filter2D(result, -1, sharpen_kernel)

                        # deblurred = cv2.fastNlMeansDenoisingColored(
                        #     sharpened, None, 10, 10, 7, 21)
                        # save frame as JPG file
                        cv2.imwrite("./ceramicimages/image"+str(count) +
                                    "_"+str(rotation)+".jpg", result)
    else:
        break
    count += 1

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()
