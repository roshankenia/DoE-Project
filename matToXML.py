from scipy.io import loadmat

# open .mat file
annots = loadmat('../SVHN/test/digitStruct.mat')
print(annots)
