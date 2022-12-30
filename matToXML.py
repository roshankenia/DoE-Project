# open .mat file
import mat73
import numpy as np
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
bboxes = digitStruct['digitStruct']['bbox']
names = digitStruct['digitStruct']['name']
for i in range(len(bboxes)):
    print('Filename:', names[i], 'BBOX:', bboxes[i])
np.save('SVHNname.npy', names)
bboxesnormal = []
for bbox in bboxes:
    heights = np.array(bbox['height'])
    labels = np.array(bbox['label'])
    lefts = np.array(bbox['left'])
    top = np.array(bbox['top'])
    width = np.array(bbox['width'])
    print(type(heights))
    digitData = []
    for j in range(len(heights)):
        xmin = lefts[i]
        ymax = top[i]

        xmax = xmin + width[i]
        ymin = ymax - heights[i]

        digitData.append((xmin, ymin, xmax, ymax, labels[i]))
        print(digitData[i])
    bboxesnormal.append(digitData)
np.save('SVHNbbox.npy', bboxesnormal)
