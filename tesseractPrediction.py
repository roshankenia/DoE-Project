import os
from PIL import Image
import pytesseract
import numpy as np
import cv2
import transforms as T
import json

pytesseract.pytesseract.tesseract_cmd = r'../../anaconda3/envs/tesseract2/bin/tesseract'
tessdata_dir_config = r'../../anaconda3/envs/tesseract2/share/tessdata'
os.environ["TESSDATA_PREFIX"] = tessdata_dir_config

# filename = './ceramicimages/image301/0.jpg'
# filename = './414.png'
# img = np.array(Image.open(filename))
# norm_img = np.zeros((img.shape[0], img.shape[1]))
# img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
# img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
# img = cv2.GaussianBlur(img, (1, 1), 0)
# text = pytesseract.image_to_string(img, config=tessdata_dir_config)

# imgNames = list(sorted(os.listdir("./trytess")))
# for imgName in imgNames:
#     img = cv2.imread(os.path.join("./trytess", imgName))
#     print("File:", imgName)
#     new_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(os.path.join("./trytess", "og_"+imgName), new_img1)
#     config = r'--oem 3 --psm 7 digits'
#     text1 = pytesseract.image_to_string(new_img1, config=config)
#     print('Result 1:', text1)
#     new_img2 = cv2.resize(new_img1, None, fx=2.5, fy=2.5,
#                           interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(os.path.join("./trytess", "scale_"+imgName), new_img2)
#     text2 = pytesseract.image_to_string(new_img2, config=config)
#     print('Result 2:', text2)
#     new_img1_ = cv2.cvtColor(new_img1, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(os.path.join("./trytess", "gray_"+imgName), new_img1_)
#     _, th1 = cv2.threshold(
#         new_img1_, 175, 255, cv2.THRESH_BINARY)
#     new_img3 = cv2.resize(th1, None, fx=2.5, fy=2.5,
#                           interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(os.path.join("./trytess", "bin_"+imgName), new_img3)
#     text3 = pytesseract.image_to_string(new_img3, config=config)
#     print('Result 3:', text3)


root = "./ceramicimages/"
pred_root = "./ceramicimagesPreds/"
if not os.path.isdir(pred_root):
    os.mkdir(pred_root)
# need to iterate through each image folder
imageFolders = list(sorted(os.listdir(root)))
config = r'--oem 3 --psm 7 digits'
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(16, 16))
for imageFolder in imageFolders:
    foldNum = ''.join(filter(lambda i: i.isdigit(), imageFolder))
    if foldNum == '300':
        # create dictionary which keeps predictions
        predDict = {}
        # obtain all image rotations
        rotations = list(sorted(os.listdir(os.path.join(root, imageFolder))))
        for rotation in rotations:
            rot = ''.join(filter(lambda i: i.isdigit(), rotation))
            print('Rotation:', rot)
            img = cv2.imread(os.path.join(root, imageFolder, rotation))
            # original image
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # obtain prediction from tesseract
            text = pytesseract.image_to_string(new_img, config=config)
            print('Result:', text)
            # increase in dictionary
            if text in predDict:
                predDict[text] += 1
            else:
                predDict[text] = 1

            # resize image
            new_img2 = cv2.resize(new_img, None, fx=2.5, fy=2.5,
                                  interpolation=cv2.INTER_CUBIC)
            text2 = pytesseract.image_to_string(new_img2, config=config)
            print('Result 2:', text2)
            if text2 in predDict:
                predDict[text2] += 1
            else:
                predDict[text2] = 1

            new_img1_ = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            # # apply CLAHE
            # img_clahe = clahe.apply(new_img1_)
            # text3 = pytesseract.image_to_string(img_clahe, config=config)
            # print('Result 3:', text3)
            # if text3 in predDict:
            #     predDict[text3] += 1
            # else:
            #     predDict[text3] = 1

            # binarized image
            _, th1 = cv2.threshold(
                new_img1_, new_img1_.max()-40, 255, cv2.THRESH_BINARY)
            new_img4 = cv2.resize(th1, None, fx=2.5, fy=2.5,
                                  interpolation=cv2.INTER_CUBIC)
            text4 = pytesseract.image_to_string(new_img4, config=config)
            print('Result 4:', text4)
            if text4 in predDict:
                predDict[text4] += 1
            else:
                predDict[text4] = 1
        # write dictionary to file
        with open(os.path.join(pred_root, "preds_"+str(foldNum)+".txt"), 'w') as file:
            file.write(json.dumps(predDict))
