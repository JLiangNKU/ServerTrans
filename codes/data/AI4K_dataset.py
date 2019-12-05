'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import time
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class AI4KDataset(data.Dataset):
    '''
    Reading the AI4K dataset for training
    Key example: XXXXXXXX_XXX_XXXXXXXX
        1st part: sequence name
        2nd part: crop index
        3rd part: frame index
    GT: Ground-Truth Frame;
    LQ: Low-Quality Frames;
    Support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(AI4KDataset, self).__init__()
        self.opt = opt

        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.
                    format(','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']

        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError('Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # keys for training and validation
        if opt['train_seqs'] and opt['valid_seqs']:
            logger.info('Loading train keys: {}'.format(opt['train_seqs']))
            logger.info('Loading valid keys: {}'.format(opt['valid_seqs']))
            self.train_seqs = pickle.load(open(opt['train_seqs'], 'rb'))  # format: XXXXXXXX
            self.valid_seqs = pickle.load(open(opt['valid_seqs'], 'rb'))  # format: XXXXXXXX
        else:
            logger.info('Using all sequences for training')

        # scene change index dictionary, key example: XXXXXXXX, value example: [0, 10, 51, 100]
        if opt['scene_index']:
            logger.info('Loading scene index: {}'.format(opt['scene_index']))
            self.scene_dict = pickle.load(open(opt['scene_index'], 'rb'))
        else:
            raise ValueError('Need to supply scene change index dictionary by running [cache_keys.py]')

        # remove the some sequences for validation
        if self.opt['phase'] == 'train':
            self.paths_GT = [v for v in self.paths_GT if v.split('_')[0] in self.train_seqs]
        elif self.opt['phase'] == 'val':
            self.paths_GT = [v for v in self.paths_GT if v.split('_')[0] in self.valid_seqs]
            self.paths_GT = self.paths_GT[::200]
        else:
            raise ValueError('Not support mode: {}'.format(self.opt['phase']))
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):

        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        seq_name, crp_name, frm_name = key.split('_')
        seq_crp_name = seq_name + '_' + crp_name
        center_frame_idx = int(frm_name) - 1  # keys originally from 1 to 100

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            # scene_list example: [0, 10, 51, 100]
            scene_list = self.scene_dict[seq_name]
            neighbor_list = []
            for i in range(len(scene_list) - 1):
                if (center_frame_idx >= scene_list[i]) and (center_frame_idx <= (scene_list[i + 1] - 1)):
                    for j in range(center_frame_idx - self.half_N_frames, center_frame_idx + self.half_N_frames + 1):
                        if j < scene_list[i]:
                            neighbor_list.append(scene_list[i])
                        elif j > (scene_list[i + 1] - 1):
                            neighbor_list.append(scene_list[i + 1] - 1)
                        else:
                            neighbor_list.append(j)
            neighbor_list = [i + 1 for i in neighbor_list]  # keys originally from 1 to 100
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            frm_name = '{:03d}'.format(neighbor_list[self.half_N_frames])
        else:
            raise NotImplementedError()

        assert len(neighbor_list) == self.opt['N_frames'],\
            'Wrong length of neighbor list: {}'.format(len(neighbor_list))

        #### get the GT image (as the center frame)
        GT_size_tuple = (3, 640, 640)  # TODO: change the size
        img_GT_path = osp.join(self.GT_root, seq_crp_name, frm_name + '.png')
        if self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, GT_size_tuple)
        else:
            img_GT = util.read_img(None, img_GT_path)

        #### get LQ images
        LQ_size_tuple = (3, 160, 160)  # TODO: change the size
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, seq_crp_name, '{:03d}.png'.format(v))
            if self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:03d}'.format(seq_crp_name, v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            LQ_size = GT_size // scale
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
            rnd_h_HR = int(rnd_h * scale)
            rnd_w_HR = int(rnd_w * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)


if __name__ == '__main__':
    pass
