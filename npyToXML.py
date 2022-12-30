import numpy as np

names = np.load('SVHNname.npy')
bboxes = np.load('SVHNbbox.npy', allow_pickle=True)

for i in range(len(names)):
    print('Filename:', names[i], "and BBOXes:", bboxes[i])
