import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pyramid_modify as pyramid
import setproctitle
from scipy import misc
import datetime
from model_pyramid import *

# tf.random.set_seed(22)
# np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--prefix', default='../day2night_modify_pyramid', type=str, help='prefix of this training')
req_grp.add_argument('--task', default='enhance', type=str, help='task to do')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--mse_lambda_low', default=0, type=float, help='weight of the mse loss.')
train_grp.add_argument('--mse_lambda_high', default=0, type=float, help='weight of the mse loss.')
train_grp.add_argument('--mse_lambda_full', default=10, type=float, help='weight of the mse loss.')
train_grp.add_argument('--tv_lambda', default=1, type=float, help='weight of the tv loss.')

train_grp.add_argument('--level', default=4, type=int, help='number of levels of the pyramid')
train_grp.add_argument('--max_iterations', default=1000000, type=int, help='training epoches')
train_grp.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--log_interval', type=int, default=100, help='interval between log messages (in s).')
train_grp.add_argument('--save_interval', type=int, default=500, help='interval between saving model checkpoints')
train_grp.add_argument('--test_interval', type=int, default=100, help='interval between testing')
train_grp.add_argument('--test_psnr_interval', type=int, default=1000, help='interval between calculating psnr')
train_grp.add_argument('--test_size', type=int, default=10, help='images to test')

data_grp = parser.add_argument_group('data pipeline')
data_grp.add_argument('--batch_size', default=1, type=int, help='size of a batch for each gradient update.')
data_grp.add_argument('--batch_size_test', default=1, type=int, help='size of a batch for each gradient update.')
data_grp.add_argument('--load_img_size', default=1024, type=int, help='size of full-size image.')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""### Load Datasets"""

if args.task == 'enhance':

    PATH = '../data/fivek_1080p'
    testA_path = os.path.join(PATH, 'big')
    print('*****trained on the image enhancement task*****')

if args.task == 'gray2color':

    PATH = '/home/liangjie/data/gray2color'
    path_file = '/home/liangjie/data/gray2color/train.txt'
    path_file_test = '/home/liangjie/data/gray2color/test.txt'
    trainA_path = os.path.join(PATH, "gray")
    trainB_path = os.path.join(PATH, "color")
    testA_path = os.path.join(PATH, 'gray_test')
    testB_path = os.path.join(PATH, 'color_test')
    print('*****trained on the gray to color translation task*****')

if args.task == 'gray2color_fivek':

    PATH = '/home/liangjie/data/gray2color_fivek_matlab'
    path_file = '/home/liangjie/data/gray2color_fivek_matlab/train.txt'
    path_file_test = '/home/liangjie/data/gray2color_fivek_matlab/test.txt'
    trainA_path = os.path.join(PATH, "gray")
    trainB_path = os.path.join(PATH, "color_trans")
    testA_path = os.path.join(PATH, 'gray_test')
    testB_path = os.path.join(PATH, 'color_test_trans')
    print('*****trained on the gray to color translation task*****')

test_all_size = len(os.listdir(testA_path))
print('test size:', test_all_size)

model_save_dir = args.prefix + '_check'
save_dir_train = args.prefix + '_train'
save_dir_test = args.prefix + '_test'

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [args.load_img_size, args.load_img_size])
    image = image * 2 - 1

    return image

test_datasetA = tf.data.Dataset.list_files(testA_path + '/*.jpg', shuffle=True)
test_datasetA = test_datasetA.shuffle(test_all_size).repeat(args.max_iterations)
test_datasetA = test_datasetA.map(lambda x: load_image(x))
test_datasetA = test_datasetA.batch(args.batch_size)
test_datasetA = test_datasetA.prefetch(args.batch_size)
test_datasetA = iter(test_datasetA)

genA2B = Generator()
transA2B = Transform_high()

genA2B_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
transA2B_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)

if os.path.exists(model_save_dir) and args.start_iter != 0:
    start_iter = args.start_iter
    genA2B.load_weights(model_save_dir + '/genA2B_' + str(start_iter))
    transA2B.load_weights(model_save_dir + '/transA2B_' + str(start_iter))
    print("Restored from checkpoints {}".format(start_iter))
else:
    print("Initializing from scratch.")


def merge_image(low_res_input, img_list):

    img_list.append(low_res_input)
    output = pyramid.merge(img_list)

    return output

def train(args, lsgan=True):
    
    multi_test_save_all = []

    for i in range(args.test_size):

        test_input = next(test_datasetA)

        multi_test_save_single = tf.reshape(tf.clip_by_value((test_input+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()

        start = time.time()

        highs_test, lows_test = pyramid.split(test_input, args.level)
        generated_low = genA2B(lows_test[-1], training=False)
        high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
        generated_high = transA2B(high_with_low_test, training=False)

        split_list_reverse_test = []

        for i_ in range(1, args.level+1):

            index = 0 - i_

            high_transed_test = tf.multiply(highs_test[index], generated_high)
            split_list_reverse_test.append(high_transed_test)
            generated_high = up_sample(generated_high)

        split_list_test = []

        index = -1

        for _ in range(args.level):

            split_list_test.append(split_list_reverse_test[index])

            index = index - 1

        generated_full = merge_image(generated_low, split_list_test)

        cost = time.time() - start

        print(cost)
        
        multi_test_save_single = np.vstack((multi_test_save_single, tf.reshape(tf.clip_by_value((generated_full+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()))
        

        if i == 0:
            multi_test_save_all = multi_test_save_single
        else:
            multi_test_save_all = np.hstack((multi_test_save_all, multi_test_save_single))

    image_path = os.path.join(save_dir_test, 'iteration_{}.jpg'.format(args.start_iter))

    misc.imsave(image_path, multi_test_save_all)


if __name__ == '__main__':

    procname = os.path.basename(args.prefix)
    setproctitle.setproctitle('train_{}'.format(procname))

    train(args, lsgan=True)
