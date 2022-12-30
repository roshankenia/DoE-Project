# open .mat file
import mat73
import numpy as np
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
bboxes = digitStruct['digitStruct']['bbox']
names = digitStruct['digitStruct']['name']
for i in range(len(bboxes)):
    print('Filename:', names[i], 'BBOX:', bboxes[i])
np.save('SVHNbbox.npy', bboxes)
np.save('SVHNname.npy', names)
