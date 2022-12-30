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
    # obtain bounding box information
    height = np.array(bbox['height'])
    label = np.array(bbox['label'])
    left = np.array(bbox['left'])
    top = np.array(bbox['top'])
    width = np.array(bbox['width'])
    print(type(height))
    print(height.size)
    digitData = []
    if height.size == 1:
        # only one bounding box
        xmin = np.float64(left)
        ymax = np.float64(top)
        xmax = xmin + np.float64(width[i])
        ymin = ymax - np.float64(height[i])
        digitData.append((xmin, ymin, xmax, ymax, np.float64(label[i])))
        print(digitData[0])
    else:
        # multiple bounding boxes found
        for j in range(len(height)):
            xmin = np.float64(left[i])
            ymax = np.float64(top[i])

            xmax = xmin + np.float64(width[i])
            ymin = ymax - np.float64(height[i])

            digitData.append((xmin, ymin, xmax, ymax, np.float64(label[i])))
            print(digitData[i])

    bboxesnormal.append(digitData)
np.save('SVHNbbox.npy', bboxesnormal)
