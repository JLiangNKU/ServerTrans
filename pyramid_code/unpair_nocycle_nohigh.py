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

# from model_pyramid import Generator, Discriminator, cycle_consistency_loss, generator_loss, discriminator_loss
from model_pyramid import *

tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--prefix', default='../day2night_modify_pyramid', type=str, help='prefix of this training')
req_grp.add_argument('--task', default='day2night', type=str, help='task to do')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=0.00001, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--recons_lambda', default=5, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--color_lambda', default=1, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--dis_lambda', default=1, type=float, help='weight of the discriminate loss.')
train_grp.add_argument('--tv_lambda', default=1, type=float, help='weight of the tv loss.')
train_grp.add_argument('--level', default=4, type=int, help='number of levels of the pyramid')
train_grp.add_argument('--max_iterations', default=1000000, type=int, help='training epoches')
train_grp.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--log_interval', type=int, default=100, help='interval between log messages (in s).')
train_grp.add_argument('--save_interval', type=int, default=500, help='interval between saving model checkpoints')
train_grp.add_argument('--test_interval', type=int, default=100, help='interval between testing')
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

"""### Load Datasets"""

if args.task == 'enhance':

    PATH = '../data/fivek_1080p'
    trainA_path = os.path.join(PATH, "input")
    trainB_path = os.path.join(PATH, "output")
    testA_path = os.path.join(PATH, 'testA')
    testB_path = os.path.join(PATH, 'testB')
    print('*****trained on the image enhancement task*****')

if args.task == 'sum2win':

    PATH = '../data/summer2winter_yosemite'
    trainA_path = os.path.join(PATH, "trainA")
    trainB_path = os.path.join(PATH, "trainB")
    testA_path = os.path.join(PATH, 'testA')
    testB_path = os.path.join(PATH, 'testB')
    print('*****trained on the summer to winter translation task*****')

if args.task == 'vangogh2photo':

    PATH = '/home/liangjie/data/vangogh2photo'
    trainA_path = os.path.join(PATH, "trainA")
    trainB_path = os.path.join(PATH, "trainB")
    testA_path = os.path.join(PATH, 'testA')
    testB_path = os.path.join(PATH, 'testB')
    print('*****trained on the vangogh2photo translation task*****')

if args.task == 'horse2zebra':

    PATH = '/home/liangjie/data/horse2zebra'
    trainA_path = os.path.join(PATH, "trainA")
    trainB_path = os.path.join(PATH, "trainB")
    testA_path = os.path.join(PATH, 'testA')
    testB_path = os.path.join(PATH, 'testB')
    print('*****trained on the horse2zebra translation task*****')

if args.task == 'cezanne2photo':

    PATH = '/home/liangjie/data/cezanne2photo'
    trainA_path = os.path.join(PATH, "trainA")
    trainB_path = os.path.join(PATH, "trainB")
    testA_path = os.path.join(PATH, 'testA')
    testB_path = os.path.join(PATH, 'testB')
    print('*****trained on the cezanne2photo translation task*****')

if args.task == 'day2night':

    PATH = '/home/liangjie/data/day2night_filtered_fine'
    trainA_path = os.path.join(PATH, "day")
    trainB_path = os.path.join(PATH, "night")
    testA_path = os.path.join(PATH, 'test_day')
    testB_path = os.path.join(PATH, 'test_night')
    print('*****trained on the day to night translation task*****')

trainA_size = len(os.listdir(trainA_path))
trainB_size = len(os.listdir(trainB_path))
print('train A:', trainA_size)
print('train B:', trainB_size)

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

train_datasetA = tf.data.Dataset.list_files(trainA_path + '/*.jpg', shuffle=True)
train_datasetA = train_datasetA.shuffle(trainA_size).repeat(args.max_iterations)
train_datasetA = train_datasetA.map(lambda x: load_image(x))
train_datasetA = train_datasetA.batch(args.batch_size)
train_datasetA = train_datasetA.prefetch(args.batch_size)
train_datasetA = iter(train_datasetA)

train_datasetB = tf.data.Dataset.list_files(trainB_path + '/*.jpg', shuffle=True)
train_datasetB = train_datasetB.shuffle(trainB_size).repeat(args.max_iterations)
train_datasetB = train_datasetB.map(lambda x: load_image(x))
train_datasetB = train_datasetB.batch(args.batch_size)
train_datasetB = train_datasetB.prefetch(args.batch_size)
train_datasetB = iter(train_datasetB)

test_datasetA = tf.data.Dataset.list_files(testA_path + '/*.jpg', shuffle=True)
test_datasetA = test_datasetA.shuffle(trainB_size).repeat(args.max_iterations)
test_datasetA = test_datasetA.map(lambda x: load_image(x))
test_datasetA = test_datasetA.batch(args.batch_size)
test_datasetA = test_datasetA.prefetch(args.batch_size)
test_datasetA = iter(test_datasetA)

a = next(train_datasetA)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

disc = Discriminator_multiscale()
gen = Generator()
# trans = Transform_high()

disc_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
gen_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
trans_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)

if os.path.exists(model_save_dir) and args.start_iter != 0:
    start_iter = args.start_iter
    disc.load_weights(model_save_dir + '/disc_' + str(start_iter))
    gen.load_weights(model_save_dir + '/gen_' + str(start_iter))
    # trans.load_weights(model_save_dir + '/trans_' + str(start_iter))
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

            genA2B_output_full = merge_image(genA2B_output, highs_A)

            disc_real_output = disc(trainB_full)

            disc_fake_output = disc(genA2B_output_full)

            disc_loss = discriminator_losssss('lsgan', disc_real_output, disc_fake_output)

            color_loss = color_losses(genA2B_output_full, trainB_full, args.batch_size)

            generator_loss = generator_losssss('lsgan', disc_fake_output)
            reconstruction_loss = tf.reduce_mean(tf.square(genA2B_output_full - trainA_full))
            tv_loss = total_variation_loss(genA2B_output_full)
            

            gen_loss = args.dis_lambda * generator_loss + args.recons_lambda * reconstruction_loss + args.color_lambda * color_loss + args.tv_lambda * tv_loss
                          
        gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
        # trans_gradients = trans_tape.gradient(gen_loss, trans.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
        disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 15)

        gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
        # trans_optimizer.apply_gradients(zip(trans_gradients, trans.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, disc.trainable_variables))

        if iteration % args.log_interval == 0:

            print('Training ' + procname + ', Iteration: {}th, Duration: {:.4f}, LOSSES: recons: {:.4f}, generator: {:.4f}, disc: {:.4f}, color: {:.4f}, tv: {:.4f}'.format(
                iteration, time.time() - start, reconstruction_loss, generator_loss.numpy(), disc_loss.numpy(), color_loss, tv_loss))

            start = time.time()

        if iteration % args.save_interval == 0:
            generate_images(trainA_full, trainA_full, genA2B_output_full, genA2B_output_full, save_dir_train, iteration)

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            # checkpoint.save(file_prefix = checkpoint_prefix)
            disc.save_weights(model_save_dir + '/disc_' + str(iteration))
            gen.save_weights(model_save_dir + '/gen_' + str(iteration))
            # trans.save_weights(model_save_dir + '/trans_' + str(iteration))
        
        if iteration % args.test_interval == 0:

            input_images = []
            output_images = []

            test_full = next(test_datasetA)

            start_test = time.time()
            
            highs_test, lows_test = pyramid.split(test_full, args.level)
            generated_low = gen(lows_test[-1], training=False)
            
            generated_full = merge_image(generated_low, highs_test)

            test_time_1 = time.time() - start_test

            input_images.append(test_full)
            output_images.append(generated_full)

            test_full = next(test_datasetA)

            start_test = time.time()
            
            highs_test, lows_test = pyramid.split(test_full, args.level)
            generated_low = gen(lows_test[-1], training=False)
            
            generated_full = merge_image(generated_low, highs_test)

            test_time_2 = time.time() - start_test

            input_images.append(test_full)
            output_images.append(generated_full)

            generate_images(input_images[0], input_images[1], output_images[0], output_images[1], save_dir_test, iteration)

            if iteration % 100 == 0:
                print('test time: ', test_time_1)
                print('test time: ', test_time_2)


if __name__ == '__main__':

    procname = os.path.basename(args.prefix)
    setproctitle.setproctitle('train_{}'.format(procname))

    train(train_datasetA, train_datasetB, args)
