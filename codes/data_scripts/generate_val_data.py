import pickle
import glob
import random
import cv2
import os
import os.path as osp
import data.util as data_util
import utils.util as util


if __name__ == '__main__':
    valid_seqs = pickle.load(open('../../keys/valid_seqs.pkl', 'rb'))

    H, W = 540, 960
    scale = 4
    LQ_size = 128
    GT_size = LQ_size * scale

    src_GT_root = '/home/data/yangxi/AI_4K/SDR_4K_PNG'
    src_LQ_root = '/home/data/yangxi/AI_4K/SDR_540p_PNG'

    dst_GT_root = '/home/ssddata/yangxi/AI_4K/SDR_4K_val'
    dst_LQ_root = '/home/ssddata/yangxi/AI_4K/SDR_540p_val'

    seq_GT_path = sorted(glob.glob(osp.join(src_GT_root, '*')))
    seq_LQ_path = sorted(glob.glob(osp.join(src_LQ_root, '*')))
    num_GT_seq = len(seq_GT_path)
    num_LQ_seq = len(seq_LQ_path)
    assert num_GT_seq == num_LQ_seq

    for i in range(num_GT_seq):
        seq_name = osp.basename(seq_GT_path[i])
        if seq_name in valid_seqs[:10]:
            print('Processing: {}'.format(seq_name))
            gt_img_list = data_util._get_paths_from_images(osp.join(src_GT_root, seq_name))
            lq_img_list = data_util._get_paths_from_images(osp.join(src_GT_root, seq_name))

            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            rnd_h_HR = int(rnd_h * scale)
            rnd_w_HR = int(rnd_w * scale)

            util.mkdir(osp.join(dst_GT_root, seq_name))
            util.mkdir(osp.join(dst_LQ_root, seq_name))

            for gt_img_path in gt_img_list:

                img_name = osp.basename(gt_img_path)
                lr_img_path = osp.join(src_LQ_root, seq_name, img_name)

                img_LQ = cv2.imread(lr_img_path, cv2.IMREAD_UNCHANGED)
                img_GT = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)

                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]

                cv2.imwrite(osp.join(dst_LQ_root, seq_name, img_name), img=img_LQ)
                cv2.imwrite(osp.join(dst_GT_root, seq_name, img_name), img=img_GT)




