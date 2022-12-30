# open .mat file
import mat73
digitStruct = mat73.loadmat('../SVHN/test/digitStruct.mat', use_attrdict=True)
print(digitStruct[300])
