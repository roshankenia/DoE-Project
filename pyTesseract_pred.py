from PIL import Image
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'../../anaconda3/envs/tesseract2/bin/tesseract.exe'

filename = './ceramicimages/image301/0.jpg'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

print(text)
