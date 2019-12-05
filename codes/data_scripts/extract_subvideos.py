import os
import os.path as osp
import sys
import cv2
import glob
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util as util


opt = {}
opt['h_crop_sz'] = 160
opt['h_step'] = 130
opt['h_thres_sz'] = 54
opt['w_crop_sz'] = 960 // 6
opt['w_step'] = 960 // 6
opt['w_thres_sz'] = 96
opt['compression_level'] = 3
opt['lr_src_folder'] = 'SDR_540p_PNG'
opt['lr_dst_folder'] = 'SDR_540p_PNG'
opt['hr_src_folder'] = 'SDR_4K_PNG'
opt['hr_dst_folder'] = 'SDR_4K_PNG'
opt['lr_crop_edge'] = 50
opt['hr_crop_edge'] = 50 * 4
opt['n_thread'] = 20
opt['salt_pepper'] = ['14022549','14798910','15738339','15853901','30672625','31545121','42579365','44575252','47470295','71724718','73623281','80109525','86174448','97752154']

img_root = '/data1/yangxi/AI_4K'
crop_root = '/data1/4KHDR/crops'
lr_src_path = osp.join(img_root, opt['lr_src_folder'])
hr_src_path = osp.join(img_root, opt['hr_src_folder'])
lr_path_list = sorted(glob.glob(lr_src_path + '/*'))
hr_path_list = sorted(glob.glob(hr_src_path + '/*'))


def extract_crops(seq_path):
    seq_name = osp.basename(seq_path)
    print('Processing: {}'.format(seq_name))
    lr_img_list = data_util._get_paths_from_images(seq_path)
    hr_img_list = data_util._get_paths_from_images(osp.join(hr_src_path, seq_name))

    # detect salt & pepper noise
    if seq_name.split('/')[-1] in opt['salt_pepper']:
        detect_salt_pepper = True
    else:
        detect_salt_pepper = False

    # detect black edges on HR image
    sample_img = cv2.imread(hr_img_list[50], cv2.IMREAD_UNCHANGED)
    sum_raws = sample_img.sum(axis=1).sum(axis=1)
    if len(np.where(sum_raws == 0)[0]) > 200:
        detect_edge = True
    else:
        detect_edge = False

    # crop LR images
    for img_path in lr_img_list:

        h_crop_sz = opt['h_crop_sz']
        h_step = opt['h_step']
        h_thres_sz = opt['h_thres_sz']
        w_crop_sz = opt['w_crop_sz']
        w_step = opt['w_step']
        w_thres_sz = opt['w_thres_sz']
        lr_crop_edge = opt['lr_crop_edge']

        img_name = osp.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if detect_salt_pepper:
            img = cv2.medianBlur(img,3)

        n_channels = len(img.shape)

        if detect_edge:
            if n_channels == 2:
                img = img[lr_crop_edge:-lr_crop_edge, :]
            else:
                img = img[lr_crop_edge:-lr_crop_edge, :, :]
            img = np.ascontiguousarray(img)

        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, h - h_crop_sz + 1, h_step)
        if h - (h_space[-1] + h_crop_sz) > h_thres_sz:
            h_space = np.append(h_space, h - h_crop_sz)

        w_space = np.arange(0, w - w_crop_sz + 1, w_step)
        if w - (w_space[-1] + w_crop_sz) > w_thres_sz:
            w_space = np.append(w_space, w - w_crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + h_crop_sz, y:y + w_crop_sz]
                else:
                    crop_img = img[x:x + h_crop_sz, y:y + w_crop_sz, :]
                crop_img = np.ascontiguousarray(crop_img)

                save_folder = osp.join(crop_root, opt['lr_dst_folder'], '{}_{:03d}'.format(seq_name, index))
                util.mkdir(save_folder)

                cv2.imwrite(
                    osp.join(save_folder, img_name[-7:]),
                    crop_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']]
                )

    # crop HR images
    for img_path in hr_img_list:

        h_crop_sz = opt['h_crop_sz'] * 4
        h_step = opt['h_step'] * 4
        h_thres_sz = opt['h_thres_sz'] * 4
        w_crop_sz = opt['w_crop_sz'] * 4
        w_step = opt['w_step'] * 4
        w_thres_sz = opt['w_thres_sz'] * 4
        hr_crop_edge = opt['lr_crop_edge'] * 4

        img_name = osp.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        n_channels = len(img.shape)

        if detect_edge:
            if n_channels == 2:
                img = img[hr_crop_edge:-hr_crop_edge, :]
            else:
                img = img[hr_crop_edge:-hr_crop_edge, :, :]
            img = np.ascontiguousarray(img)

        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, h - h_crop_sz + 1, h_step)
        if h - (h_space[-1] + h_crop_sz) > h_thres_sz:
            h_space = np.append(h_space, h - h_crop_sz)

        w_space = np.arange(0, w - w_crop_sz + 1, w_step)
        if w - (w_space[-1] + w_crop_sz) > w_thres_sz:
            w_space = np.append(w_space, w - w_crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + h_crop_sz, y:y + w_crop_sz]
                else:
                    crop_img = img[x:x + h_crop_sz, y:y + w_crop_sz, :]
                crop_img = np.ascontiguousarray(crop_img)

                save_folder = osp.join(crop_root, opt['hr_dst_folder'], '{}_{:03d}'.format(seq_name, index))
                util.mkdir(save_folder)

                cv2.imwrite(
                    osp.join(save_folder, img_name[-7:]),
                    crop_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']]
                )

def main(mode='single'):
    if mode == 'single':
        for seq_path in lr_path_list[4::5]:
            extract_crops(seq_path)
    else:
        pool = Pool(opt['n_thread'])
        for seq_path in lr_path_list:
            pool.apply_async(extract_crops, args=(seq_path))
        pool.close()
        pool.join()
        print('All subprocesses done.')

if __name__ == '__main__':
    main(mode='single')
