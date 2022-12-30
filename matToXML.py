# open .mat file
import mat73
import numpy as np
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
# Save
np.save('digitStruct.npy', digitStruct['digitStruct'])

# Load
digitStruct = np.load('digitStruct.npy', allow_pickle='TRUE').item()
for keys, value in digitStruct.items():
    print(keys)
