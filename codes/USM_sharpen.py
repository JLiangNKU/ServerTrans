import os
import os.path as osp
import sys
import glob
import numpy as np
import cv2
from PIL import Image, ImageFilter
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util 
import utils.util as util
from torchvision import transforms
from skimage.filters import unsharp_mask
import skimage.io as io
from skimage import img_as_ubyte

black_edge = [0,70,70,70,70,70,70,0,70,70,
              0,0,70,0,69,70,69,70,52,70,
              70,70,70,69,0,0,70,0,69,0,
              70,52,0,69,52,52,52,70,0,70,
              69,0,69,70,0,52,52,0,70,70]

img_root = '../results/AI4K'
save_folder = '../results/sharpen'
path_list = sorted(glob.glob(img_root + '/*'))

sharp_filter = 'gaussian'

seq_id = 0
step = 1
for path in path_list[seq_id::step]:
    seq_name = osp.basename(path)
    util.mkdir(osp.join(save_folder, seq_name))
    print('Processing: {}'.format(seq_name))
    lr_img_list = data_util._get_paths_from_images(path)
    for img_path in lr_img_list:
        img_name = osp.basename(img_path)
        
        if sharp_filter == 'gaussian':
            im = Image.open(img_path)
            im2 = im.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=3))
            imT = transforms.ToTensor()(im)
            imT2 = transforms.ToTensor()(im2)
            edge = black_edge[seq_id] * 4
            if edge > 0:
                imT2[:,edge-3:edge+3,:] = imT[:,edge-3:edge+3,:]
                imT2[:, 2160 - edge - 3:2160-edge + 3, :] = imT[:, 2160 - edge - 3:2160 - edge + 3, :]
            im2 = transforms.ToPILImage()(imT2)
            im2.save(osp.join(save_folder, seq_name, img_name))
            
        elif sharp_filter == 'bilateral':
            im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            bilateral = cv2.bilateralFilter(im, 11, 500, 3)
            diff = im - bilateral
            result = im + diff * 4.0
            edge = black_edge[seq_id] * 4
            m = 4
            if edge > 0:
                result[edge - m:edge + m,:, :] = im[edge - m:edge + m, :, :]
                result[2160 - edge - m:2160-edge + m, :, :] = im[2160 - edge - m:2160 - edge + m, :, :]
            cv2.imwrite(osp.join(save_folder, seq_name, img_name), result,[cv2.IMWRITE_PNG_COMPRESSION, 3])

    seq_id += step