# open .mat file
import mat73
import numpy as np
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
bboxes = digitStruct['digitStruct']['bbox']
names = digitStruct['digitStruct']['name']

np.save('SVHNname.npy', names)
bboxesnormal = []
for bbox in bboxes:
    print()
    # obtain bounding box information
    height = np.array(bbox['height'])
    label = np.array(bbox['label'])
    left = np.array(bbox['left'])
    top = np.array(bbox['top'])
    width = np.array(bbox['width'])

    digitData = []
    if height.size == 1:
        # only one bounding box
        xmin = np.float64(left)
        ymin = np.float64(top)
        xmax = xmin + np.float64(width)
        ymax = ymin + np.float64(height)
        digitData.append([xmin, ymin, xmax, ymax, np.float64(label)])
        print(digitData[0])
    else:
        # multiple bounding boxes found
        for j in range(len(height)):
            xmin = np.float64(left[j])
            ymin = np.float64(top[j])

            xmax = xmin + np.float64(width[j])
            ymax = ymin + np.float64(height[j])

            digitData.append([xmin, ymin, xmax, ymax, np.float64(label[j])])
            print(digitData[j])

    bboxesnormal.append(digitData)
np.save('SVHNbbox.npy', bboxesnormal)
