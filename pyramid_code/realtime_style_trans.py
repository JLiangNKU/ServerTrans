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

tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--prefix', default='../cityscapes', type=str, help='prefix of this training')
req_grp.add_argument('--task', default='cityscapes', type=str, help='task to do')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--mse_lambda_low', default=0, type=float, help='weight of the mse loss.')
train_grp.add_argument('--mse_lambda_high', default=0, type=float, help='weight of the mse loss.')
train_grp.add_argument('--mse_lambda_full', default=1, type=float, help='weight of the mse loss.')
train_grp.add_argument('--color_lambda', default=0, type=float, help='weight of the tv loss.')
train_grp.add_argument('--tv_lambda', default=0, type=float, help='weight of the tv loss.')

train_grp.add_argument('--level', default=2, type=int, help='number of levels of the pyramid')
train_grp.add_argument('--max_iterations', default=1000000, type=int, help='training epoches')
train_grp.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--log_interval', type=int, default=100, help='interval between log messages (in s).')
train_grp.add_argument('--save_interval', type=int, default=500, help='interval between saving model checkpoints')
train_grp.add_argument('--test_interval', type=int, default=100, help='interval between testing')
train_grp.add_argument('--test_psnr_interval', type=int, default=1000, help='interval between calculating psnr')
train_grp.add_argument('--test_size', type=int, default=6, help='images to test')

data_grp = parser.add_argument_group('data pipeline')
data_grp.add_argument('--batch_size', default=1, type=int, help='size of a batch for each gradient update.')
data_grp.add_argument('--batch_size_test', default=1, type=int, help='size of a batch for each gradient update.')
data_grp.add_argument('--load_img_size', default=256, type=int, help='size of full-size image.')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""### Load Datasets"""

if args.task == 'cityscapes':

    PATH = '/home/liangjie/data/cityscapes'
    trainA_path = os.path.join(PATH, "train")
    testA_path = os.path.join(PATH, 'test')
    print('*****trained on the gray to color translation task*****')


train_size = len(os.listdir(trainA_path))
print('train size:', train_size)

test_all_size = len(os.listdir(testA_path))
print('test size:', test_all_size)

model_save_dir = args.prefix + '_check'
save_dir_train = args.prefix + '_train'
save_dir_test = args.prefix + '_test'


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [args.load_img_size, args.load_img_size * 2])
    image = image * 2 - 1

    return image

train_datasetA = tf.data.Dataset.list_files(trainA_path + '/*.*', shuffle=True)
train_datasetA = train_datasetA.shuffle(train_size).repeat(args.max_iterations)
train_datasetA = train_datasetA.map(lambda x: load_image(x))
train_datasetA = train_datasetA.batch(args.batch_size)
train_datasetA = train_datasetA.prefetch(args.batch_size)
train_datasetA = iter(train_datasetA)

test_datasetA = tf.data.Dataset.list_files(testA_path + '/*.*', shuffle=True)
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

    if not os.path.exists(save_dir_test):
        os.mkdir(save_dir_test)
    
    the_time = datetime.datetime.now()
    logs = open(save_dir_test + '/' + 'train_process.txt', "a")
    logs.write('#' * 20)
    logs.write('\n')
    logs.write(str(the_time))
    logs.write('\n')
    logs.write('#' * 20)
    logs.write('\n')
    logs.write(str(args))
    logs.write('\n\n')
    logs.close()

    start = time.time()

    for iteration in range(args.start_iter, args.max_iterations):

        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as transA2B_tape:
            try:
                 train_full = next(train_dataset)
                 trainA_full, trainB_full = train_full[:, :, 256:, :], train_full[:, :, :256, :]

            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            highs_A, lows_A = pyramid.split(trainA_full, args.level)
            highs_B, lows_B = pyramid.split(trainB_full, args.level)
            
            genA2B_output = genA2B(lows_A[-1], training=True)

            high_with_low_A = tf.concat([highs_A[-1], lows_A[-2]], 3)

            highA2B_output_o = transA2B(high_with_low_A, training=True)

            highA2B_output = highA2B_output_o

            split_list_A_reverse = []

            for i in range(1, args.level+1):

                index = 0 - i

                high_transed_A = tf.multiply(highs_A[index], highA2B_output)
                split_list_A_reverse.append(high_transed_A)
                if i < args.level:
                    highA2B_output = up_sample(highA2B_output)

            split_list_A = []

            index = -1

            for _ in range(args.level):

                split_list_A.append(split_list_A_reverse[index])

                index = index - 1

            genA2B_output_full = merge_image(genA2B_output, split_list_A)

            # mse_loss_full = tf.reduce_mean(tf.square(tf.clip_by_value((trainB_full+1)/2, 0, 1) - tf.clip_by_value((genA2B_output_full+1)/2, 0, 1)))
            # mse_loss_low = tf.reduce_mean(tf.square(tf.clip_by_value((lows_B[-1]+1)/2, 0, 1) - tf.clip_by_value((genA2B_output+1)/2, 0, 1)))
            # mse_loss_high = tf.reduce_mean(tf.square(tf.clip_by_value((highs_B[0]+1)/2, 0, 1) - tf.clip_by_value((split_list_A[0]+1)/2, 0, 1)))

            mse_loss_full = tf.reduce_mean(tf.square(trainB_full - genA2B_output_full))
            mse_loss_low = tf.reduce_mean(tf.square(lows_B[-1] - genA2B_output))
            mse_loss_high = tf.reduce_mean(tf.square(highs_B[0] - split_list_A[0]))

            # color_loss = color_losses(genA2B_output_full, trainB_full, args.batch_size)

            # tv_loss = total_variation_loss(tf.clip_by_value((genA2B_output_full+1)/2, 0, 1))

            gen_loss = args.mse_lambda_low * mse_loss_low + args.mse_lambda_full * mse_loss_full # + args.tv_lambda * tv_loss + args.color_lambda * color_loss# 
            trans_loss = args.mse_lambda_full * mse_loss_full + args.mse_lambda_high * mse_loss_high # + args.tv_lambda * tv_loss + args.color_lambda * color_loss# 
            
        genA2B_gradients = genA2B_tape.gradient(gen_loss, genA2B.trainable_variables)
        transA2B_gradients = transA2B_tape.gradient(trans_loss, transA2B.trainable_variables)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))
        transA2B_optimizer.apply_gradients(zip(transA2B_gradients, transA2B.trainable_variables))

        if iteration % args.log_interval == 0:

            generated = np.clip((genA2B_output_full+1)/2, 0, 1)
            target = np.clip((trainB_full+1)/2, 0, 1)

            # generated = generated[0, 100:900, 100:900, :]
            # target = target[0, 100:900, 100:900, :]

            psnr = tf.image.psnr(generated, target, max_val=1.0)

            logs_train = 'Training ' + procname + ', Iteration: {}th, time: {:.4f}, LOSSES: mse_low: {:.4f}, mse_high: {:.4f}, mse_full: {:.4f}, psnr: {:.4f}'.format(
                iteration, time.time() - start, mse_loss_low, mse_loss_high, mse_loss_full, psnr[0])

            print(logs_train)

            logs = open(save_dir_test + '/' + 'train_process.txt', "a")
            logs.write(logs_train)
            logs.write('\n')
            logs.close()

            start = time.time()

        if iteration % args.save_interval == 0:

            # generate_images(trainA_full[0], trainB_full[0], genA2B_output_full[0], genA2B_output_full[0], save_dir_train, iteration)

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

            genA2B.save_weights(model_save_dir + '/genA2B_' + str(iteration))
            transA2B.save_weights(model_save_dir + '/transA2B_' + str(iteration))
        
        if iteration % args.test_interval == 0:

            multi_test_save_all = []

            for i in range(args.test_size):

                test_full = next(test_datasetA)
                test_input, test_output = test_full[:, :, 256:, :], test_full[:, :, :256, :]

                multi_test_save_single = tf.reshape(tf.clip_by_value((test_input+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()

                # start_test = time.time()
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
                
                multi_test_save_single = np.vstack((multi_test_save_single, tf.reshape(tf.clip_by_value((generated_full+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()))
                multi_test_save_single = np.vstack((multi_test_save_single, tf.reshape(tf.clip_by_value((test_output+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()))

                if i == 0:
                    multi_test_save_all = multi_test_save_single
                else:
                    multi_test_save_all = np.hstack((multi_test_save_all, multi_test_save_single))

            image_path = os.path.join(save_dir_test, 'iteration_{}.jpg'.format(iteration))

            misc.imsave(image_path, multi_test_save_all)

        if iteration % args.test_psnr_interval == 0 and iteration != 0:

            psnr_all = []
            cost_all = []

            print('******************************')
            print('testing on all test images...')

            for i in range(test_all_size):

                test_full = next(test_datasetA)
                test_input, test_output = test_full[:, :, 256:, :], test_full[:, :, :256, :]

                start = time.time()

                highs_test, lows_test = pyramid.split(test_input, args.level)
                generated_low = genA2B(lows_test[-1], training=False)
                high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
                generated_high = transA2B(high_with_low_test, training=False)

                split_list_reverse_test = []

                for i in range(1, args.level+1):

                    index = 0 - i

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

                generated = np.clip((generated_full+1)/2, 0, 1)
                target = np.clip((test_output+1)/2, 0, 1)

                # generated = generated[0, 100:900, 100:900, :]
                # target = target[0, 100:900, 100:900, :]

                psnr_test = tf.image.psnr(generated, target, max_val=1.0)

                psnr_all.append(psnr_test[0].numpy())
                cost_all.append(cost)

            logs_test = 'test iteration: {}th, mean psnr: {:.4f}, avg inference time: {:.4f}'.format(iteration, np.mean(psnr_all), np.mean(cost_all))
            print(logs_test)
            print('****************************************************************************')
            logs = open(save_dir_test + '/' + 'train_process.txt', "a")
            logs.write(logs_test)
            logs.write('\n')
            logs.close()

if __name__ == '__main__':

    procname = os.path.basename(args.prefix)
    setproctitle.setproctitle('train_{}'.format(procname))

    train(train_datasetA, args, lsgan=True)
