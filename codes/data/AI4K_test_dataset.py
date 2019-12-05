import pickle
import os.path as osp
import torch
import torch.utils.data as data
import data.util as util


class AI4KTestdataset(data.Dataset):
    """
    Reading the AI4K dataset for testing
    """

    def __init__(self, opt):
        super(AI4KTestdataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': []}

        if self.cache_data == False:
            raise ValueError('Currently only support cache_data=True.')

        if self.data_type == 'lmdb':
            raise ValueError('Currently not support LMDB during validation/test.')

        # scene change index dictionary, key example: XXXXXXXX, value example: [0, 10, 51, 100]
        if opt['scene_index']:
            self.scene_dict = pickle.load(open(opt['scene_index'], 'rb'))
        else:
            raise ValueError('Need to supply scene change index dictionary by running [cache_keys.py]')

        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_GT)  # subfolders_name: XXXXXXXX
            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)
            max_idx = len(img_paths_LQ)
            assert len(img_paths_LQ) == len(img_paths_GT), 'Different number of images in LQ and GT folders'
            self.data_info['path_LQ'].extend(img_paths_LQ)
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        if self.cache_data:
            select_idx = util.index_generation_with_scene_list(idx, max_idx, self.opt['N_frames'],
                                                               self.scene_dict[folder],
                                                               padding=self.opt['padding'])
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]

        return {'LQs': imgs_LQ, 'GT': img_GT, 'folder': folder, 'idx': self.data_info['idx'][index]}

    def __len__(self):
        return len(self.data_info['path_GT'])


if __name__ == '__main__':

    opt = {}
    opt['name'] = 'AI4K'
    opt['dataroot_LQ'] = '/home/ssddata/yangxi/AI_4K/SDR_540p_val'
    opt['dataroot_GT'] = '/home/ssddata/yangxi/AI_4K/SDR_4K_val'
    opt['scene_index'] = '../../keys/scene_idx.pkl'
    opt['cache_data'] = True
    opt['data_type'] = 'img'
    opt['N_frames'] = 5
    opt['padding'] = 'replicate'

    dataset = AI4KTestdataset(opt)