import os
from PIL import Image
import pytesseract
import numpy as np
import cv2
pytesseract.pytesseract.tesseract_cmd = r'../../anaconda3/envs/tesseract2/bin/tesseract'
tessdata_dir_config = r'../../anaconda3/envs/tesseract2/share/tessdata'
os.environ["TESSDATA_PREFIX"] = tessdata_dir_config

# filename = './ceramicimages/image301/0.jpg'
filename = './tess.jpg'
img = np.array(Image.open(filename))
norm_img = np.zeros((img.shape[0], img.shape[1]))
img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)
text = pytesseract.image_to_string(img, config=tessdata_dir_config)

print('Result:', text)
