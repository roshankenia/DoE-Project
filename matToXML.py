# open .mat file
import mat73
import numpy as np
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
np.save('../SVHN/test/svhndict.npy', digitStruct)

# Load
digitStruct = np.load('svhndict.npy', allow_pickle='TRUE').item()
print(digitStruct['name'])
