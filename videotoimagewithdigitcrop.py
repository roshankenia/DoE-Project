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
pebble_model = torch.load('../../DoE-Project/mask-rcnn-pebble.pt')
pebble_model.eval()
CLASS_NAMES = ['__background__', 'pebble']
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
pebble_model.to(device)

digits_model = torch.load(r'./saved_model/digits_detector.pkl')
digits_model.to(device)


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


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
    pred = pebble_model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    print('pred scores:', pred_score)
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
    # cv2.imwrite("./ceramicimages/image"+str(ind) + "_mask.jpg", img)


def crop_pebble(img, masks, boxes, ind):
    mask = np.asarray(masks[0], dtype="uint8")
    # obtain only the mask pixels from the image
    only_mask = cv2.bitwise_and(img, img, mask=mask)
    bbox = boxes[0]
    # crop the image to only contain the pebble
    crop = only_mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # put pebble on standard 1000x1000 image
    imgSize = 1100
    background = np.zeros((imgSize, imgSize, 3), np.uint8)
    ch, cw = crop.shape[:2]

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((imgSize-ch)/2)
    xoff = round((imgSize-cw)/2)

    background[yoff:yoff+ch, xoff:xoff+cw] = crop
    # save crop as JPG file
    # cv2.imwrite("./ceramicimages/image" + str(ind) + "/crop.jpg", background)

    return background


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


def create_overlap_box(bbox1, bbox2):
    # check if these two bboxes overlap
    xmin1 = bbox1[0]
    ymin1 = bbox1[1]
    xmax1 = bbox1[2]
    ymax1 = bbox1[3]

    xmin2 = bbox2[0]
    ymin2 = bbox2[1]
    xmax2 = bbox2[2]
    ymax2 = bbox2[3]
    print(xmin1, ymin1, xmax1, ymax1)
    print(xmin2, ymin2, xmax2, ymax2)

    overlaps = True
    # if rectangle has area 0, no overlap
    if xmin1 == xmax1 or ymin1 == ymax1 or xmin2 == xmax2 or ymin2 == ymax2:
        overlaps = False
        print('Zero Area')
    # If one rectangle is on left side of other
    if xmin1 > xmax2 or xmin2 > xmax1:
        overlaps = False
        print('X Bad')
    # If one rectangle is above other
    if ymax1 < ymin2 or ymax2 < ymin1:
        overlaps = False
        print('Y Bad')
    if overlaps:
        print('\n\nFOUND OVERLAPPING')
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
    print('RETURNING REGULAR')
    return [bboxes[0], bboxes[1], bboxes[2]]


def create_digit_crops(model, img):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    # image_array = img.numpy()
    image_array = np.array(normalize(img), dtype=np.float32)
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

    bboxes = prediction[0]['boxes'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    print(bboxes)
    goodBBoxes = []
    for i in range(len(scores)):
        if scores[i] >= 0.4:
            goodBBoxes.append(bboxes[i])
        else:
            # scores already sorted
            break

    # for i in range(prediction[0]['boxes'].cpu().shape[0]): # select all the predicted bounding boxes
    newBBoxes = goodBBoxes
    if len(goodBBoxes) >= 3:
        # combine those that overlap
        newBBoxes = create_new_bboxes(goodBBoxes)
    elif len(goodBBoxes) == 2:
        # check if two overlap
        combined = create_overlap_box(goodBBoxes[0], goodBBoxes[1])
        if combined != None:
            newBBoxes = [combined]
    # draw boxes if they exist
    if(len(newBBoxes) != 0):
        # create digit crops
        digitCrops = []
        for i in range(len(newBBoxes)):
            bbox = newBBoxes[i]
            digits_crop = img[round(bbox[1]):round(
                bbox[3]), round(bbox[0]):round(bbox[2])]
            digitCrops.append(digits_crop)
        return digitCrops


vidcap = cv2.VideoCapture(
    '../../DoE-Project/Moving Pebbles - Ceramic Paint.MOV')
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('video has', str(frame_count), 'frames.')

rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
             165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

count = 0
while (vidcap.isOpened()):
    hasFrames, image = vidcap.read()
    if hasFrames:
        # check if image has a pebble
        masks, boxes, pred_cls = get_prediction(image, .9)
        if masks is not None:
            if len(masks) == 1:
                # make image directory
                path = "./ceramicimages/image" + str(count) + "/"
                if not os.path.isdir(path):
                    os.mkdir(path)
                    # save unmodified image
                # save frame as JPG file
                cv2.imwrite(path + "unmodified.jpg", image)
                # make_mask_image(np.copy(image), masks,
                #                 boxes, pred_cls, count)
                image = crop_pebble(np.copy(image), masks, boxes, count)

                # now try to obtain digit crop
                digits_crop = create_digit_crops(digits_model, image)
                for c in range(len(digits_crop)):
                    digit_crop = digits_crop[c]
                    cv2.imwrite(path + "digit_crop_"+str(c)+".jpg", digit_crop)
                # resize image
                # image = cv2.resize(image, (100, 100))

                # for rotation in rotations:
                #     image_center = tuple(np.array(image.shape[1::-1]) / 2)
                #     rot_mat = cv2.getRotationMatrix2D(
                #         image_center, rotation, 1.0)
                #     result = cv2.warpAffine(
                #         image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                #     # sharpened = cv2.filter2D(result, -1, sharpen_kernel)

                #     # deblurred = cv2.fastNlMeansDenoisingColored(
                #     #     sharpened, None, 10, 10, 7, 21)
                #     # save frame as JPG file
                #     cv2.imwrite(path + str(rotation)+".jpg", result)
    else:
        break
    count += 1

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()
