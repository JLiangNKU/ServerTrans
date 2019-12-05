import cv2
import numpy as np
 
for i in range(5000):

    print(i+1)

    img=cv2.imread(r'/home/liangjie/1code/pyramid/data/fivek_1080p/input/' + str(i+1) + '.jpg')# 读入图片
    img2=cv2.imread(r'/home/liangjie/1code/pyramid/data/fivek_1080p/output/' + str(i+1) + '.jpg')

    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (1024, 1024), interpolation=cv2.INTER_CUBIC)


    concated_img=np.concatenate((img,img2),axis=1)

    cv2.imwrite('/home/liangjie/1code/pyramid/data/fivek_1080p/concated/' + str(i+1) + '.jpg', concated_img)