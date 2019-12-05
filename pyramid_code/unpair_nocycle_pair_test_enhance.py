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
from model_pyramid_enhance import *

tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--prefix', default='../enhance_recons5', type=str, help='prefix of this training')
req_grp.add_argument('--task', default='enhance', type=str, help='task to do')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--recons_lambda', default=5, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--color_lambda', default=0, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--dis_lambda', default=1, type=float, help='weight of the discriminate loss.')
train_grp.add_argument('--tv_lambda', default=0, type=float, help='weight of the tv loss.')
train_grp.add_argument('--level', default=4, type=int, help='number of levels of the pyramid')
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
data_grp.add_argument('--load_img_size', default=1024, type=int, help='size of full-size image.')

args = parser.parse_args()
print(args)

AUTOTUNE = tf.data.experimental.AUTOTUNE

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

if args.task == 'gray2color_fivek':

    PATH = '/home/liangjie/data/gray2color_fivek_matlab'
    path_file = '/home/liangjie/data/gray2color_fivek_matlab/train.txt'
    path_file_test = '/home/liangjie/data/gray2color_fivek_matlab/test.txt'
    trainA_path = os.path.join(PATH, "gray")
    trainB_path = os.path.join(PATH, "color_trans")
    testA_path = os.path.join(PATH, 'gray_test')
    testB_path = os.path.join(PATH, 'color_test_trans')
    print('*****trained on the gray to color translation task*****')

if args.task == 'gray2color':

    PATH = '/home/liangjie/data/gray2color'
    path_file = '/home/liangjie/data/gray2color/train.txt'
    path_file_test = '/home/liangjie/data/gray2color/test.txt'
    trainA_path = os.path.join(PATH, "gray")
    trainB_path = os.path.join(PATH, "color")
    testA_path = os.path.join(PATH, 'gray_test')
    testB_path = os.path.join(PATH, 'color_test')
    print('*****trained on the gray to color translation task*****')

trainA_size = len(os.listdir(trainA_path))
print('trainA size:', trainA_size)
trainB_size = len(os.listdir(trainB_path))
print('trainB size:', trainB_size)

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

def load_and_preprocess_from_path_label(path1, path2):
    return load_image(path1), load_image(path2)

train_datasetA = tf.data.Dataset.list_files(trainA_path + '/*.*', shuffle=True)
train_datasetA = train_datasetA.shuffle(trainA_size).repeat(args.max_iterations)
train_datasetA = train_datasetA.map(lambda x: load_image(x))
train_datasetA = train_datasetA.batch(args.batch_size)
train_datasetA = train_datasetA.prefetch(args.batch_size)
train_datasetA = iter(train_datasetA)

train_datasetB = tf.data.Dataset.list_files(trainB_path + '/*.*', shuffle=True)
train_datasetB = train_datasetB.shuffle(trainB_size).repeat(args.max_iterations)
train_datasetB = train_datasetB.map(lambda x: load_image(x))
train_datasetB = train_datasetB.batch(args.batch_size)
train_datasetB = train_datasetB.prefetch(args.batch_size)
train_datasetB = iter(train_datasetB)

with open(path_file_test, 'r') as fid:
    flist_test = [l.strip() for l in fid.readlines()]

input_files_test = [os.path.join(testA_path, f) for f in flist_test]
output_files_test = [os.path.join(testB_path, f) for f in flist_test]

test_dataset = tf.data.Dataset.from_tensor_slices((input_files_test, output_files_test))
test_dataset = test_dataset.shuffle(500).repeat()
test_dataset = test_dataset.map(load_and_preprocess_from_path_label)
test_dataset = test_dataset.batch(args.batch_size_test)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = iter(test_dataset)

# disc = Discriminator_multiscale()
disc = Discriminator()
gen = Generator()
trans = Transform_high()

disc_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
gen_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
trans_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)

if os.path.exists(model_save_dir) and args.start_iter != 0:
    start_iter = args.start_iter
    disc.load_weights(model_save_dir + '/disc_' + str(start_iter))
    gen.load_weights(model_save_dir + '/gen_' + str(start_iter))
    trans.load_weights(model_save_dir + '/trans_' + str(start_iter))
    print("Restored from checkpoints {}".format(start_iter))
else:
    print("Initializing from scratch.")

def generate_images(A, B, A2B, B2A, save_dir, epoch):

    plt.figure(figsize=(25, 25))
    A = tf.reshape(A, [args.load_img_size, args.load_img_size, 3]).numpy()
    B = tf.reshape(B, [args.load_img_size, args.load_img_size, 3]).numpy()
    B2A = tf.reshape(B2A, [args.load_img_size, args.load_img_size, 3]).numpy()
    A2B = tf.reshape(A2B, [args.load_img_size, args.load_img_size, 3]).numpy()
    display_list = [A, B, A2B, B2A]

    if 'test' in save_dir:
        title = ['inputA', 'inputB', 'outputA2B', 'outputB2A']
    else:
        title = ['A', 'B', 'A2B', 'B2A']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        plt.imshow(np.clip((display_list[i]+1)/2, 0, 1))
        # plt.imshow((display_list[i]+1)/2)
        plt.axis('off')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.savefig(save_dir + '/generated_%d.png'%epoch)
    plt.close()

def merge_image(low_res_input, img_list):

    img_list.append(low_res_input)
    output = pyramid.merge(img_list)

    return output

def train(train_datasetA, train_datasetB, args):

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

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as trans_tape:
            try:
                trainA_full = next(train_datasetA)
                trainB_full = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            highs_A, lows_A = pyramid.split(trainA_full, args.level)
            
            genA2B_output = gen(lows_A[-1], training=True)

            high_with_low_A = tf.concat([highs_A[-1], lows_A[-2]], 3)

            mask_0, mask_1, mask_2, mask_3 = trans(high_with_low_A, training=True)

            split_list_A = highs_A

            split_list_A[-1] = tf.multiply(highs_A[-1], mask_0)
            split_list_A[-2] = tf.multiply(highs_A[-2], mask_1)
            split_list_A[-3] = tf.multiply(highs_A[-3], mask_2)
            split_list_A[-4] = tf.multiply(highs_A[-4], mask_3)

            genA2B_output_full = merge_image(genA2B_output, split_list_A)

            # genA2B_output_full = tf.clip_by_value((genA2B_output_full+1)/2, 0, 1)
            # trainA_full = tf.clip_by_value((trainA_full+1)/2, 0, 1)
            # trainB_full = tf.clip_by_value((trainB_full+1)/2, 0, 1)

            disc_real_output = disc(trainB_full)

            disc_fake_output = disc(genA2B_output_full)

            disc_loss = discriminator_loss_cal(disc_real_output, disc_fake_output)
            generator_loss = generator_loss_cal(disc_fake_output)

            # generator_loss = generator_losssss('gan', disc_fake_output)
            # disc_loss = discriminator_losssss('gan', disc_real_output, disc_fake_output)

            # color_loss = color_losses(genA2B_output_full, trainB_full, args.batch_size)

            
            reconstruction_loss = tf.reduce_mean(tf.square(genA2B_output_full - trainA_full))
            # tv_loss = total_variation_loss(genA2B_output_full)
            
            gen_loss = args.dis_lambda * generator_loss + args.recons_lambda * reconstruction_loss #+ args.color_lambda * color_loss + args.tv_lambda * tv_loss

        gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
        trans_gradients = trans_tape.gradient(gen_loss, trans.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
        # disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 15)

        gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
        trans_optimizer.apply_gradients(zip(trans_gradients, trans.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, disc.trainable_variables))

        if iteration % args.log_interval == 0:

            logs_train = 'Training ' + procname + ', Iteration: {}th, Duration: {:.4f}, LOSSES: recons: {:.4f}, generator: {:.4f}, disc: {:.4f}'.format(
                iteration, time.time() - start, reconstruction_loss, generator_loss.numpy(), disc_loss.numpy())

            print(logs_train)

            logs = open(save_dir_test + '/' + 'train_process.txt', "a")
            logs.write(logs_train)
            logs.write('\n')
            logs.close()

            start = time.time()

        if iteration % args.save_interval == 0:
            # generate_images(trainA_full, trainA_full, genA2B_output_full, highA_output, save_dir_train, iteration)

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            # checkpoint.save(file_prefix = checkpoint_prefix)
            disc.save_weights(model_save_dir + '/disc_' + str(iteration))
            gen.save_weights(model_save_dir + '/gen_' + str(iteration))
            trans.save_weights(model_save_dir + '/trans_' + str(iteration))
        
        if iteration % args.test_interval == 0:

            multi_test_save_all = []

            for i in range(args.test_size):

                test_input, test_output = next(test_dataset)

                multi_test_save_single = tf.reshape(tf.clip_by_value((test_input+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()

                # start_test = time.time()
                highs_test, lows_test = pyramid.split(test_input, args.level)
                generated_low = gen(lows_test[-1], training=False)
                high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
                mask_0, mask_1, mask_2, mask_3 = trans(high_with_low_test, training=True)

                split_list_test = highs_test

                split_list_test[-1] = tf.multiply(highs_test[-1], mask_0)
                split_list_test[-2] = tf.multiply(highs_test[-2], mask_1)
                split_list_test[-3] = tf.multiply(highs_test[-3], mask_2)
                split_list_test[-4] = tf.multiply(highs_test[-4], mask_3)

                generated_full = merge_image(generated_low, split_list_test)
                
                multi_test_save_single = np.vstack((multi_test_save_single, tf.reshape(tf.clip_by_value((generated_full+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()))
                multi_test_save_single = np.vstack((multi_test_save_single, tf.reshape(tf.clip_by_value((test_output+1)/2, 0, 1), [args.load_img_size, args.load_img_size, 3]).numpy()))

                if i == 0:
                    multi_test_save_all = multi_test_save_single
                else:
                    multi_test_save_all = np.hstack((multi_test_save_all, multi_test_save_single))

            image_path = os.path.join(save_dir_test, 'iteration_{}.jpg'.format(iteration))

            misc.imsave(image_path, multi_test_save_all)

        if iteration % args.test_psnr_interval == 0:

            psnr_all = []
            cost_all = []

            print('******************************')
            print('testing on all test images...')

            for i in range(test_all_size):

                test_input, test_output = next(test_dataset)

                start = time.time()

                highs_test, lows_test = pyramid.split(test_input, args.level)
                generated_low = gen(lows_test[-1], training=False)
                high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
                mask_0, mask_1, mask_2, mask_3 = trans(high_with_low_test, training=True)

                split_list_test = highs_test

                split_list_test[-1] = tf.multiply(highs_test[-1], mask_0)
                split_list_test[-2] = tf.multiply(highs_test[-2], mask_1)
                split_list_test[-3] = tf.multiply(highs_test[-3], mask_2)
                split_list_test[-4] = tf.multiply(highs_test[-4], mask_3)

                generated_full = merge_image(generated_low, split_list_test)

                cost = time.time() - start

                generated = tf.clip_by_value((generated_full+1)/2, 0, 1)
                target = tf.clip_by_value((test_output+1)/2, 0, 1)

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

    train(train_datasetA, train_datasetB, args)
