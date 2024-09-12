import cv2
import os
import numpy as np
sem_path = os.path.join('/home/lzh/3dscene/data/MonoScene/NYU/NYU_dataset/depthbin/NYUtrain/', "sem_map", "NYU0004_0000.png")
myimag = cv2.imread(sem_path, -1)#.astype(int)
sem_path1 = os.path.join('/home/lzh/3dscene/JetsonSSD/NYU_images/train/', "000004_category_suncg.png")
myimag1 = cv2.imread(sem_path1, -1)#.astype(int)
sem_path2 = os.path.join('/home/lzh/3dscene/JetsonSSD/NYU_images/train/', "000004_category_suncg_new.png")
myimag2 = cv2.imread(sem_path2, -1)#.astype(int)
myimag = myimag[None, :, :]
myimag = np.concatenate((myimag, myimag, myimag), axis=0)
cv2.imshow('IMREAD_COLOR+Color', myimag)
cv2.waitKey()
