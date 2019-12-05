'''
Test AI4K datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import pickle

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
seq_name = ['16536366','16842928','18523430','18964850','22921150','23662872','26287421','26455661','33207731','33454400',
            '39167513','39234459','42954477','43091004','46382743','46393768','49104192','49586309','54003182','54513097',
            '56459353','56990135','61883118','62108829','65342200','66042767','68778359','68847251','74146494','74532233',
            '75912281','76244369','77044862','77413330','80910417','82497910','84357217','85355224','89332027','92059330',
            '93834608','94544607','94924039','95616624','96096191','96251032','96325943','97714559','98472411','99176273']
crop_edge = [0,70,70,70,70,70,70,0,70,70,
             0,0,70,0,69,70,69,70,52,70,
             70,70,70,69,0,0,70,0,69,0,
             70,52,0,69,52,52,52,70,0,70,
             69,0,69,70,0,52,52,0,70,70]

def main(gpu_id,start_id,step):
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    data_mode = 'AI4K'

    stage = 1  # 1 or 2
    flip_test = True

    #### model
    if data_mode == 'AI4K':
        if stage == 1:
            model_path = '/home/zenghui/projects/4KHDR/experiments/pretrained_models/EDVR_L_G300k.pth'       # TODO: change path
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'  # TODO: change path
    else:
        raise NotImplementedError()

    N_in = 5  # use N_in images to restore one HR image
    predeblur, HR_in = False, False
    back_RBs = 40
    if stage == 2:
        HR_in = True
        back_RBs = 20
    model = EDVR_arch.EDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### dataset
    if data_mode == 'AI4K':
        test_dataset_folder = '../datasets/SDR_540p_PNG_test'  # TODO: change path
    else:
        raise NotImplementedError()

    #### scene information
    scene_index_path = '../keys/test_scene_idx.pkl'  # TODO: change path
    scene_dict = pickle.load(open(scene_index_path, 'rb'))

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    padding = 'replicate' # temporal padding mode
    save_imgs = True
    save_folder = '../results_edvr_l_tsa_300k_2/{}'.format(data_mode)  # TODO: change path
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []
    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    seq_id = start_id
    for subfolder in subfolder_l[start_id::step]:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        logger.info('Processing sequence: {}, seq_id = {}, crop_edge = {}'.format(subfolder_name,seq_id,crop_edge[seq_id]))
        hr_crop_edge = crop_edge[seq_id]*4

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))

        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ images
        imgs_LQ = data_util.read_img_seq(subfolder)

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation_with_scene_list(img_idx, max_idx, N_in,
                                                                    scene_dict[subfolder_name], padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                output = util.single_forward(model, imgs_in)
            #if crop_edge[seq_id]>0:
            #    output[:,:,:hr_crop_edge, :] = 0
            #    output[:,:,-hr_crop_edge:, :] = 0
            output = util.tensor2img(output.squeeze(0))

            
            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

        seq_id += step 


if __name__ == '__main__':
    main(gpu_id='4',start_id=4,step=5)
