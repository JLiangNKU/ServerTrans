# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import time
 
 
# img = cv2.imread('/home/ljj/1code/xxx.jpg')
 
# start = time.time()
# dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# print('time: ', time.time() - start)

# plt.subplot(121),plt.imshow(img)
# plt.subplot(122),plt.imshow(dst)
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture('/home/ljj/1code/13926303.mp4')

# create a list of first 5 frames
img = [cap.read()[1] for i in range(5)]

# Denoise 3rd frame considering all the 5 frames
start = time.time()
dst = cv2.fastNlMeansDenoisingColoredMulti(img, 2, 5, None, 4, 7, 35)
print('time: ', time.time() - start)

# plt.subplot(131),plt.imshow(img[2],'gray')
# plt.subplot(133),plt.imshow(dst,'gray')
# plt.show()

plt.imshow(img[2],'gray')
plt.savefig('ori.jpg')
plt.imshow(dst,'gray')
plt.savefig('after.jpg')