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


def make_mask_image(img, ind, rect_th=2, text_size=2, text_th=2):
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(2000*px, 1100*px))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    vis_tgt_path = "./ceramicimages/"
    if not os.path.isdir(vis_tgt_path):
        os.mkdir(vis_tgt_path)
    plt.savefig(os.path.join(
        vis_tgt_path, "image" + str(ind) + "_mask.jpg"))
    plt.close()
    # save frame as JPG file
    cv2.imwrite("./ceramicimages/image"+str(count) + "_usingCV.jpg", img)


vidcap = cv2.VideoCapture('Moving Pebbles - Ceramic Paint.MOV')
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('video has', str(frame_count), 'frames.')

rotations = [0, 45, 90, 135, 180, 225, 270, 315]
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

count = 0
while (vidcap.isOpened()):
    hasFrames, image = vidcap.read()
    if hasFrames:
        if count == 1950:
            # check if image has a pebble
            masks, boxes, pred_cls = get_prediction(image, .9)
            if masks is not None:
                make_mask_image(np.copy(image), count)
            # save unmodified image
            # save frame as JPG file
            cv2.imwrite("./ceramicimages/image" +
                        str(count) + "_unmodified.jpg", image)
            for rotation in rotations:
                image_center = tuple(np.array(image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(
                    image_center, rotation, 1.0)
                result = cv2.warpAffine(
                    image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                sharpened = cv2.filter2D(result, -1, sharpen_kernel)

                deblurred = cv2.fastNlMeansDenoisingColored(
                    sharpened, None, 10, 10, 7, 21)
                # save frame as JPG file
                cv2.imwrite("./ceramicimages/image"+str(count) +
                            "_"+str(rotation)+".jpg", deblurred)
    else:
        break
    count += 1

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()
