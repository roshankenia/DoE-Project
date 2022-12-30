# open .mat file
import mat73
data_dict = mat73.loadmat('../SVHN/test/digitStruct.mat')
for keys, value in data_dict.items():
    print(keys)
