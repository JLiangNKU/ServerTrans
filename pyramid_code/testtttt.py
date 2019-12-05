"""
# translate the color image into the gray one

import cv2
import os

path = '/home/liangjie/data/gray2color_fivek/color_test'
path_save = '/home/liangjie/data/gray2color_fivek/color_test_trans'

for name in os.listdir(path):

    print(name)

    img = cv2.imread(os.path.join(path, name))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.join(path_save, name), img)

"""


# create the txt file indicating the name of files

import os

path = '/home/liangjie/data/fivek/expert/original'
path_save = '/home/liangjie/data/fivek'

for file_name in os.listdir(path):
    with open(os.path.join(path_save, 'train_original.txt'), 'a') as f:
        if int(file_name[1:5]) <= 4500:
            f.write(file_name+'\n')

