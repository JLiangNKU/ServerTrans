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

from model_pyramid_enhance import *

tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--prefix', default='../colorization', type=str, help='prefix of this training')
req_grp.add_argument('--task', default='gray2color_fivek', type=str, help='task to do')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--mse_lambda_low', default=0, type=float, help='weight of the mse loss.')
train_grp.add_argument('--mse_lambda_full', default=10, type=float, help='weight of the mse loss.')
train_grp.add_argument('--tv_lambda', default=1, type=float, help='weight of the tv loss.')

train_grp.add_argument('--level', default=4, type=int, help='number of levels of the pyramid')
train_grp.add_argument('--max_iterations', default=1000000, type=int, help='training epoches')
train_grp.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--log_interval', type=int, default=100, help='interval between log messages (in s).')
train_grp.add_argument('--save_interval', type=int, default=500, help='interval between saving model checkpoints')
train_grp.add_argument('--test_interval', type=int, default=100, help='interval between testing')
train_grp.add_argument('--test_psnr_interval', type=int, default=1000, help='interval between calculating psnr')
train_grp.add_argument('--test_size', type=int, default=2, help='images to test')

data_grp = parser.add_argument_group('data pipeline')
data_grp.add_argument('--batch_size', default=1, type=int, help='size of a batch for each gradient update.')
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
    path_file = '../data/fivek_1080p/filelist_fivek_train.txt'
    path_file_test = '../data/fivek_1080p/filelist_fivek_test.txt'
    trainA_path = os.path.join(PATH, "input")
    trainB_path = os.path.join(PATH, "output")
    testA_path = os.path.join(PATH, 'test_input')
    testB_path = os.path.join(PATH, 'test_output')
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

train_size = len(os.listdir(trainA_path))
print('train size:', train_size)

test_size = len(os.listdir(testA_path))
print('test size:', test_size)

model_save_dir = args.prefix + '_check'
save_dir_train = args.prefix + '_train'
save_dir_test = args.prefix + '_test'

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [args.load_img_size, args.load_img_size])
    image = image * 2 - 1
    return image

def load_and_preprocess_from_path_label(path1, path2):
    return load_and_preprocess_image(path1), load_and_preprocess_image(path2)

with open(path_file, 'r') as fid:
    flist_train = [l.strip() for l in fid.readlines()]

input_files = [os.path.join(trainA_path, f) for f in flist_train]
output_files = [os.path.join(trainB_path, f) for f in flist_train]

train_dataset = tf.data.Dataset.from_tensor_slices((input_files, output_files))
train_dataset = train_dataset.shuffle(train_size).repeat()
train_dataset = train_dataset.map(load_and_preprocess_from_path_label)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
train_dataset = iter(train_dataset)

with open(path_file_test, 'r') as fid:
    flist_test = [l.strip() for l in fid.readlines()]

input_files_test = [os.path.join(testA_path, f) for f in flist_test]
output_files_test = [os.path.join(testB_path, f) for f in flist_test]

test_dataset = tf.data.Dataset.from_tensor_slices((input_files_test, output_files_test))
test_dataset = test_dataset.shuffle(500).repeat()
test_dataset = test_dataset.map(load_and_preprocess_from_path_label)
test_dataset = test_dataset.batch(args.batch_size)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = iter(test_dataset)

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

def generate_images(A, B, B2A, A2B, save_dir, epoch):

    plt.figure(figsize=(25, 25))
    A = tf.reshape(A, [args.load_img_size, args.load_img_size, 3]).numpy()
    B = tf.reshape(B, [args.load_img_size, args.load_img_size, 3]).numpy()
    B2A = tf.reshape(B2A, [args.load_img_size, args.load_img_size, 3]).numpy()
    A2B = tf.reshape(A2B, [args.load_img_size, args.load_img_size, 3]).numpy()
    display_list = [A, B, B2A, A2B]

    if 'test' in save_dir:
        title = ['input1', 'input2', 'output1', 'output2']
    else:
        title = ['input', 'target', 'output', 'input+target']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        plt.imshow(np.clip((display_list[i]+1)/2, 0, 1))
        plt.axis('off')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.savefig(save_dir + '/generated_%d.png'%epoch)
    plt.close()


def merge_image(low_res_input, img_list):

    img_list.append(low_res_input)
    output = pyramid.merge(img_list)

    return output

def train(train_dataset, args, lsgan=True):
    

    start = time.time()

    for iteration in range(args.start_iter, args.max_iterations):

        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as transA2B_tape:
            try:
                trainA_full, trainB_full = next(train_dataset)

            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            highs_A, lows_A = pyramid.split(trainA_full, args.level)
            highs_B, lows_B = pyramid.split(trainB_full, args.level)

            # print('max pixel value_input: ', trainA.numpy().mean())
            # print('min pixel value_input: ', trainA.numpy().std())
            # print('max pixel value_target: ', trainB.numpy().mean())
            # print('min pixel value_target: ', trainB.numpy().std())
            # print('*********************************')
            
            genA2B_output = genA2B(lows_A[-1], training=True)

            high_with_low_A = tf.concat([highs_A[-1], lows_A[-2]], 3)

            mask_0, mask_1, mask_2, mask_3 = transA2B(high_with_low_A, training=True)

            split_list_A = highs_A

            split_list_A[-1] = tf.multiply(highs_A[-1], mask_0)
            split_list_A[-2] = tf.multiply(highs_A[-2], mask_1)
            split_list_A[-3] = tf.multiply(highs_A[-3], mask_2)
            split_list_A[-4] = tf.multiply(highs_A[-4], mask_3)

            genA2B_output_full = merge_image(genA2B_output, split_list_A)

            mse_loss_full = tf.reduce_mean(tf.square(trainB_full - genA2B_output_full))
            mse_loss_low = tf.reduce_mean(tf.square(lows_B[-1] - genA2B_output))

            # color_loss = color_losses(genA2B_output_full, trainB_full, args.batch_size)

            # tv_loss = total_variation_loss(genA2B_output_full)

            genA2B_loss = args.mse_lambda_low * mse_loss_low + args.mse_lambda_full * mse_loss_full# + color_loss# + args.tv_lambda * tv_loss
            
        genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)
        transA2B_gradients = transA2B_tape.gradient(genA2B_loss, transA2B.trainable_variables)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))
        transA2B_optimizer.apply_gradients(zip(transA2B_gradients, transA2B.trainable_variables))

        if iteration % args.log_interval == 0:

            generated = np.clip((genA2B_output_full+1)/2, 0, 1)
            target = np.clip((trainB_full+1)/2, 0, 1)

            psnr = tf.image.psnr(generated, target, max_val=1.0)

            print('Training ' + procname + ', Iteration: {}th, time: {:.4f}, LOSSES: mse_low: {:.4f}, mse_full: {:.4f}, psnr: {:.4f}'.format(
                iteration, time.time() - start, mse_loss_low, mse_loss_full, psnr[0]))
            start = time.time()

        if iteration % args.save_interval == 0:

            generate_images(trainA_full, trainB_full, genA2B_output_full, genA2B_output_full, save_dir_train, iteration)

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

            genA2B.save_weights(model_save_dir + '/genA2B_' + str(iteration))
            transA2B.save_weights(model_save_dir + '/transA2B_' + str(iteration))
        
        if iteration % args.test_interval == 0:

            input_images = []
            # target_images = []
            output_images = []

            for i in range(args.test_size):

                test_input, test_output = next(test_dataset)

                # start_test = time.time()
                highs_test, lows_test = pyramid.split(test_input, args.level)
                generated_low = genA2B(lows_test[-1], training=False)
                high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
                mask_0, mask_1, mask_2, mask_3 = transA2B(high_with_low_test, training=True)

                split_list_test = highs_test

                split_list_test[-1] = tf.multiply(highs_test[-1], mask_0)
                split_list_test[-2] = tf.multiply(highs_test[-2], mask_1)
                split_list_test[-3] = tf.multiply(highs_test[-3], mask_2)
                split_list_test[-4] = tf.multiply(highs_test[-4], mask_3)

                generated_full = merge_image(generated_low, split_list_test)

                # test_time = time.time() - start_test

                # if iteration % 100 == 0:

                #     generated = np.clip((generated_full+1)/2, 0, 1)
                #     target = np.clip((test_output+1)/2, 0, 1)

                #     psnr_test = tf.image.psnr(generated, target, max_val=1.0)
                #     print('test time: {:.4f}, psnr: {:.4f}'.format(test_time, psnr_test[0]))

                input_images.append(test_input)
                output_images.append(generated_full)
                
                # image_shape = np.shape(generated_full)
                
                # saved_image = np.zeros([image_shape[1], image_shape[2]*3, image_shape[3]])
                # saved_image[:, image_shape[1]*0 : image_shape[1]*1, :] = (test_input[0, :, :, :])# * 255.0
                # saved_image[:, image_shape[1]*1 : image_shape[1]*2, :] = (generated_full[0, :, :, :])# * 255.0
                # saved_image[:, image_shape[1]*2 : image_shape[1]*3, :] = (test_output[0, :, :, :])# * 255.0
                # # print(saved_image.shape)

                # if not os.path.exists(args.save_dir_test):
                #     os.mkdir(args.save_dir_test)

                # save_path = args.save_dir_test + '/generated_{}_{}.png'.format(iteration, i)

                # skimage.io.imsave(save_path, saved_image)

            generate_images(input_images[0], input_images[1], output_images[0], output_images[1], save_dir_test, iteration)

        if (iteration + 1) % args.test_psnr_interval == 0:

            psnr_all = []
            cost_all = []

            print('******************************')
            print('testing on all test images...')

            for i in range(test_size):

                test_input, test_output = next(test_dataset)

                start = time.time()

                highs_test, lows_test = pyramid.split(test_input, args.level)
                generated_low = genA2B(lows_test[-1], training=False)
                high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
                mask_0, mask_1, mask_2, mask_3 = transA2B(high_with_low_test, training=True)

                split_list_test = highs_test

                split_list_test[-1] = tf.multiply(highs_test[-1], mask_0)
                split_list_test[-2] = tf.multiply(highs_test[-2], mask_1)
                split_list_test[-3] = tf.multiply(highs_test[-3], mask_2)
                split_list_test[-4] = tf.multiply(highs_test[-4], mask_3)

                generated_full = merge_image(generated_low, split_list_test)

                cost = time.time() - start

                generated = np.clip((generated_full+1)/2, 0, 1)
                target = np.clip((test_output+1)/2, 0, 1)

                psnr_test = tf.image.psnr(generated, target, max_val=1.0)

                psnr_all.append(psnr_test[0].numpy())
                cost_all.append(cost)

            print('test iteration: {}th, mean psnr: {:.4f}, mean inference time: {:.4f}'.format(iteration, np.mean(psnr_all), np.mean(cost_all)))
            print('****************************************************************************')

if __name__ == '__main__':

    procname = os.path.basename(args.prefix)
    setproctitle.setproctitle('train_{}'.format(procname))

    train(train_dataset, args, lsgan=True)
