import os
import os.path as osp
import glob
import pickle
import random
import pandas as pd
import data.util as util
from collections import Counter


def save_keys(save_path, root):
    path_list, _ = util.get_image_paths('img', root)
    keys_list = []
    for path in path_list:
        name_list = path.split('/')
        keys_list.append(name_list[-2] + '_' + name_list[-1].split('.')[0])
    with open(save_path, 'wb') as f:
        pickle.dump({'keys': keys_list}, f)


def load_keys(save_path):
    with open(save_path, 'rb') as f:
        keys_list = pickle.load(f)['keys']
    return keys_list


def train_valid_split(root,
                      num_train_seq, num_valid_seq,
                      save_train_list=False, save_valid_list=False,
                      save_path_train_list=None, save_path_valid_list=None):
    '''split the 700 sequences for training and validation'''
    seq_list = sorted(glob.glob(root))
    seq_list = [osp.basename(seq) for seq in seq_list]
    num_total_seq = len(seq_list)
    assert num_train_seq + num_valid_seq <= num_total_seq
    print('Total number of sequences: ', num_total_seq)
    print('Number of sequences for training: ', num_train_seq)
    print('Number of sequences for validation: ', num_valid_seq)
    # shuffle the sequence list
    random.shuffle(seq_list)
    train_seq = seq_list[:num_train_seq]
    valid_seq = seq_list[-num_valid_seq:]
    if save_train_list:
        with open(save_path_train_list, 'wb') as f:
            pickle.dump(train_seq, f)
    if save_valid_list:
        with open(save_path_valid_list, 'wb') as f:
            pickle.dump(valid_seq, f)
    return train_seq, valid_seq


def scene_index(root, verbose=False, save_dict=False, save_path=None):
    '''get the scene change index for all the 700 sequences'''
    scene_count = Counter()
    csv_path_list = sorted(glob.glob(root))
    start_dict = {}

    for csv_path in csv_path_list:
        seq_name = osp.basename(csv_path).split('-')[0]
        csv = pd.read_csv(csv_path, skiprows=[0])
        start = []
        for i in range(len(csv)):
            start.append(csv.loc[i, 'Start Frame'])
        scene_count[len(start)] += 1
        start.append(100)
        start_dict[seq_name] = start
        if verbose:
            print(csv_path, len(start))

    if save_dict:
        with open(save_path, 'wb') as f:
            pickle.dump(start_dict, f)

    return start_dict, scene_count


if __name__ == '__main__':

    # # example
    root = '/home/ssddata/4KHDR/crops/SDR_540p_PNG'  # crop root
    save_path = '../../keys/AI4K_keys.pkl'
    save_keys(save_path, root)
    keys_list = load_keys(save_path)

    # # example
    root = '/home/hddata2/4KHDR/images/SDR_540p_PNG/*'  # img root
    train_seq, valid_seq = train_valid_split(root, num_train_seq=690, num_valid_seq=10,
                                             save_train_list=True, save_valid_list=True,
                                             save_path_train_list='../../keys/train_seqs.pkl',
                                             save_path_valid_list='../../keys/valid_seqs.pkl')

    # # example
    # root = '/home/xiyang/Datasets/4KHDR/video_scenes_thres35/*'
    # start_dict, scene_count = scene_index(root, verbose=True, save_dict=True, save_path='../../keys/scene_idx.pkl')

