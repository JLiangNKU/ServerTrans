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
train_grp.add_argument('--cyc_lambda', default=5, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--color_lambda', default=1, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--dis_lambda', default=1, type=float, help='weight of the discriminate loss.')

train_grp.add_argument('--level', default=5, type=int, help='number of levels of the pyramid')
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

if args.task == 'mysum2win':

    PATH = '/home/liangjie/data/summer2winter'
    trainA_path = os.path.join(PATH, "summer")
    trainB_path = os.path.join(PATH, "winter")
    testA_path = os.path.join(PATH, 'test_summer')
    testB_path = os.path.join(PATH, 'test_winter')
    print('*****trained on the summer to winter translation task*****')

if args.task == 'mygray2color':

    PATH = '/home/liangjie/data/gray2color'
    trainA_path = os.path.join(PATH, "gray")
    trainB_path = os.path.join(PATH, "color")
    testA_path = os.path.join(PATH, 'gray_test')
    testB_path = os.path.join(PATH, 'color_test')
    print('*****trained on the gray to color translation task*****')

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

test_datasetA = tf.data.Dataset.list_files(testA_path + '/*.*', shuffle=True)
test_datasetA = test_datasetA.shuffle(trainB_size).repeat(args.max_iterations)
test_datasetA = test_datasetA.map(lambda x: load_image(x))
test_datasetA = test_datasetA.batch(args.batch_size)
test_datasetA = test_datasetA.prefetch(args.batch_size)
test_datasetA = iter(test_datasetA)

test_datasetB = tf.data.Dataset.list_files(testB_path + '/*.*', shuffle=True)
test_datasetB = test_datasetB.shuffle(trainB_size).repeat(args.max_iterations)
test_datasetB = test_datasetB.map(lambda x: load_image(x))
test_datasetB = test_datasetB.batch(args.batch_size)
test_datasetB = test_datasetB.prefetch(args.batch_size)
test_datasetB = iter(test_datasetB)

a = next(train_datasetA)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

discA = Discriminator_multiscale()
discB = Discriminator_multiscale()
genA2B = Generator()
genB2A = Generator()
transA2B = Transform_high()
transB2A = Transform_high()

discA_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
discB_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
genA2B_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
genB2A_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
transA2B_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
transB2A_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)

if os.path.exists(model_save_dir) and args.start_iter != 0:
    start_iter = args.start_iter
    discA.load_weights(model_save_dir + '/discA_' + str(start_iter))
    discB.load_weights(model_save_dir + '/discB_' + str(start_iter))
    genA2B.load_weights(model_save_dir + '/genA2B_' + str(start_iter))
    genB2A.load_weights(model_save_dir + '/genB2A_' + str(start_iter))
    transA2B.load_weights(model_save_dir + '/transA2B_' + str(start_iter))
    transB2A.load_weights(model_save_dir + '/transB2A_' + str(start_iter))
    print("Restored from checkpoints {}".format(start_iter))
else:
    print("Initializing from scratch.")

# checkpoint_dir = args.model_save_dir
# checkpoint_prefix = os.path.join(checkpoint_dir, args.task)
# checkpoint = tf.train.Checkpoint(discA_optimizer=discA_optimizer,
#                                  discB_optimizer=discB_optimizer,
#                                  genA2B_optimizer=genA2B_optimizer,
#                                  genB2A_optimizer=genB2A_optimizer,
#                                  discA=discA,
#                                  discB=discB,
#                                  genA2B=genA2B,
#                                  genB2A=genB2A)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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

        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape, tf.GradientTape() as transA2B_tape, tf.GradientTape() as transB2A_tape:
            try:
                trainA_full = next(train_datasetA)
                trainB_full = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            highs_A, lows_A = pyramid.split(trainA_full, args.level)
            highs_B, lows_B = pyramid.split(trainB_full, args.level)
            
            genA2B_output = genA2B(lows_A[-1], training=True)
            genB2A_output = genB2A(lows_B[-1], training=True)

            high_with_low_A = tf.concat([highs_A[-1], lows_A[-2]], 3)
            high_with_low_B = tf.concat([highs_B[-1], lows_B[-2]], 3)

            # kernel = pyramid._binomial_kernel(tf.shape(input=lows_A[-1])[3], dtype=lows_A[-1].dtype)

            # low_up_before_A = pyramid._upsample(lows_A[-1], kernel)
            # low_up_after_A = pyramid._upsample(genA2B_output, kernel)
            # low_up_before_B = pyramid._upsample(lows_B[-1], kernel)
            # low_up_after_B = pyramid._upsample(genB2A_output, kernel)

            # high_with_low_A = tf.concat([highs_A[-1], low_up_before_A], 3)
            # high_with_low_B = tf.concat([highs_B[-1], low_up_before_B], 3)

            # high_with_low_A = tf.concat([high_with_low_A, low_up_after_A], 3)
            # high_with_low_B = tf.concat([high_with_low_B, low_up_after_B], 3)

            highA2B_output_o = transA2B(high_with_low_A, training=True)
            highB2A_output_o = transB2A(high_with_low_B, training=True)

            highA2B_output = highA2B_output_o
            highB2A_output = highB2A_output_o

            split_list_A_reverse = []
            split_list_B_reverse = []

            for i in range(1, args.level+1):

                index = 0 - i

                high_transed_A = tf.multiply(highs_A[index], highA2B_output)
                split_list_A_reverse.append(high_transed_A)
                highA2B_output = up_sample(highA2B_output)

                high_transed_B = tf.multiply(highs_B[index], highB2A_output)
                split_list_B_reverse.append(high_transed_B)
                highB2A_output = up_sample(highB2A_output)

            split_list_A = []
            split_list_B = []

            index = -1

            for _ in range(args.level):

                split_list_A.append(split_list_A_reverse[index])
                split_list_B.append(split_list_B_reverse[index])

                index = index - 1

            genA2B_output_full = merge_image(genA2B_output, split_list_A)
            genB2A_output_full = merge_image(genB2A_output, split_list_B)

            reconstructed_A_low = genB2A(genA2B_output, training=True)
            reconstructed_B_low = genA2B(genB2A_output, training=True)

            #################

            high_with_low_A_r = tf.concat([highA2B_output_o, lows_A[-2]], 3)
            high_with_low_B_r = tf.concat([highB2A_output_o, lows_B[-2]], 3)

            highA2B_output_r = transB2A(high_with_low_A_r, training=True)
            highB2A_output_r = transA2B(high_with_low_B_r, training=True)

            split_list_A_reverse = []
            split_list_B_reverse = []

            for i in range(1, args.level+1):

                index = 0 - i

                high_transed_A = tf.multiply(highs_A[index], highA2B_output_r)
                split_list_A_reverse.append(high_transed_A)
                highA2B_output_r = up_sample(highA2B_output_r)

                high_transed_B = tf.multiply(highs_B[index], highB2A_output_r)
                split_list_B_reverse.append(high_transed_B)
                highB2A_output_r = up_sample(highB2A_output_r)

            split_list_A = []
            split_list_B = []

            index = -1

            for _ in range(args.level):

                split_list_A.append(split_list_A_reverse[index])
                split_list_B.append(split_list_B_reverse[index])

                index = index - 1

            #################

            reconstructed_A_full = merge_image(reconstructed_A_low, split_list_A)
            reconstructed_B_full = merge_image(reconstructed_B_low, split_list_B)

            discA_real_output = discA(trainA_full)
            discB_real_output = discB(trainB_full)

            discA_fake_output = discA(genB2A_output_full)
            discB_fake_output = discB(genA2B_output_full)

            discA_loss = discriminator_losssss('lsgan', discA_real_output, discA_fake_output)
            discB_loss = discriminator_losssss('lsgan', discB_real_output, discB_fake_output)

            color_loss_A2B = color_losses(genA2B_output_full, trainB_full, args.batch_size)
            color_loss_B2A = color_losses(genB2A_output_full, trainA_full, args.batch_size)

            # content_loss_A2B = content_losses(genA2B_output_full, trainA_full, args.batch_size)
            # content_loss_B2A = content_losses(genB2A_output_full, trainB_full, args.batch_size)

            # gp_A = gradient_penalty(trainA_full, genB2A_output_full, discA)
            # gp_B = gradient_penalty(trainB_full, genA2B_output_full, discB)

            generatorA2B_loss = generator_losssss('lsgan', discB_fake_output)
            cycleA2B_loss = cycle_consistency_loss(trainA_full, trainB_full, reconstructed_A_full, reconstructed_B_full)
            
            generatorB2A_loss = generator_losssss('lsgan', discA_fake_output)
            cycleB2A_loss = cycle_consistency_loss(trainA_full, trainB_full, reconstructed_A_full, reconstructed_B_full)
            

            genA2B_loss = args.dis_lambda * generatorA2B_loss + args.cyc_lambda * cycleA2B_loss + args.color_lambda * color_loss_A2B
                          
            genB2A_loss = args.dis_lambda * generatorB2A_loss + args.cyc_lambda * cycleB2A_loss + args.color_lambda * color_loss_B2A
                          

        genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)
        genB2A_gradients = genB2A_tape.gradient(genB2A_loss, genB2A.trainable_variables)
        transA2B_gradients = transA2B_tape.gradient(genA2B_loss, transA2B.trainable_variables)
        transB2A_gradients = transB2A_tape.gradient(genB2A_loss, transB2A.trainable_variables)

        discA_gradients = discA_tape.gradient(discA_loss, discA.trainable_variables)
        discA_gradients, _ = tf.clip_by_global_norm(discA_gradients, 15)
        discB_gradients = discB_tape.gradient(discB_loss, discB.trainable_variables)
        discB_gradients, _ = tf.clip_by_global_norm(discB_gradients, 15)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))
        genB2A_optimizer.apply_gradients(zip(genB2A_gradients, genB2A.trainable_variables))
        transA2B_optimizer.apply_gradients(zip(transA2B_gradients, transA2B.trainable_variables))
        transB2A_optimizer.apply_gradients(zip(transB2A_gradients, transB2A.trainable_variables))

        discA_optimizer.apply_gradients(zip(discA_gradients, discA.trainable_variables))
        discB_optimizer.apply_gradients(zip(discB_gradients, discB.trainable_variables))

        if iteration % args.log_interval == 0:

            print('Training ' + procname + ', Iteration: {}th, Duration: {:.4f}, LOSSES: cycle: {:.4f}, generator: {:.4f}, disc: {:.4f}, color: {:.4f}'.format(
                iteration, time.time() - start, cycleA2B_loss, generatorA2B_loss.numpy(), discA_loss.numpy(), color_loss_A2B))

            start = time.time()

        if iteration % args.save_interval == 0:
            generate_images(trainA_full, trainB_full, genA2B_output_full, genB2A_output_full, save_dir_train, iteration)

            # print('Time taken for iteration {} is {} sec'.format(iteration + 1, time.time() - start))

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            # checkpoint.save(file_prefix = checkpoint_prefix)
            discA.save_weights(model_save_dir + '/discA_' + str(iteration))
            discB.save_weights(model_save_dir + '/discB_' + str(iteration))
            genA2B.save_weights(model_save_dir + '/genA2B_' + str(iteration))
            genB2A.save_weights(model_save_dir + '/genB2A_' + str(iteration))  
            transA2B.save_weights(model_save_dir + '/transA2B_' + str(iteration))
            transB2A.save_weights(model_save_dir + '/transB2A_' + str(iteration))  
        
        if iteration % args.test_interval == 0:

            input_images = []
            output_images = []

            test_full = next(test_datasetA)

            start_test = time.time()
            
            highs_test, lows_test = pyramid.split(test_full, args.level)
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

            test_time_1 = time.time() - start_test

            input_images.append(test_full)
            output_images.append(generated_full)

            test_full = next(test_datasetB)

            start_test = time.time()
            
            highs_test, lows_test = pyramid.split(test_full, args.level)
            generated_low = genB2A(lows_test[-1], training=False)
            high_with_low_test = tf.concat([highs_test[-1], lows_test[-2]], 3)
            generated_high = transB2A(high_with_low_test, training=False)

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
