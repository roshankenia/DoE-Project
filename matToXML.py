# open .mat file
import mat73
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
for keys, value in digitStruct['digitStruct'].items():
    print(keys)
