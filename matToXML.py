# open .mat file
import mat73
data_dict = mat73.loadmat('../SVHN/test/digitStruct.mat')
print(data_dict)
