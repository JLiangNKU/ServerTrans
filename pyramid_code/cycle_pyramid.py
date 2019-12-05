import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pyramid
import setproctitle

from model_pyramid import Generator, Discriminator, cycle_consistency_loss, generator_loss, discriminator_loss

# tf.random.set_seed(22)
# np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--model_save_dir', default='../cycle_check', type=str, help='directory to save checkpoints to.')
req_grp.add_argument('--task', default='sum2win', type=str, help='task to do')
req_grp.add_argument('--save_dir_train', default='../cycle_train', type=str, help=' ')
req_grp.add_argument('--save_dir_test', default='../cycle_test', type=str, help=' ')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--cyc_lambda', default=10, type=float, help='weight of the cycle loss.')
train_grp.add_argument('--dis_lambda', default=1, type=float, help='weight of the dis loss.')
train_grp.add_argument('--level', default=4, type=int, help='number of levels of the pyramid')
train_grp.add_argument('--max_iterations', default=1000000, type=int, help='training epoches')
train_grp.add_argument('--gpu_id', default='0', type=str, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--log_interval', type=int, default=100, help='interval between log messages (in s).')
train_grp.add_argument('--save_interval', type=int, default=500, help='interval between saving model checkpoints')
train_grp.add_argument('--test_interval', type=int, default=500, help='interval between testing')
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
    test_path = os.path.join(PATH, 'test')
    print('*****trained on the image enhancement task*****')

if args.task == 'sum2win':

    PATH = '../data/summer2winter_yosemite'
    trainA_path = os.path.join(PATH, "trainA")
    trainB_path = os.path.join(PATH, "trainB")
    test_path = os.path.join(PATH, 'testA')
    print('*****trained on the summer to winter translation task*****')

if args.task == 'day2night':

    PATH = '/home/liangjie/data/day2night_filtered'
    trainA_path = os.path.join(PATH, "day")
    trainB_path = os.path.join(PATH, "night")
    test_path = os.path.join(PATH, 'test')
    print('*****trained on the day to night translation task*****')

trainA_size = len(os.listdir(trainA_path))
trainB_size = len(os.listdir(trainB_path))
print('train A:', trainA_size)
print('train B:', trainB_size)

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [args.load_img_size, args.load_img_size])
    # image = (image / 127.5) - 1
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

test_dataset = tf.data.Dataset.list_files(test_path + '/*.jpg', shuffle=True)
test_dataset = test_dataset.shuffle(trainB_size).repeat(args.max_iterations)
test_dataset = test_dataset.map(lambda x: load_image(x))
test_dataset = test_dataset.batch(args.batch_size)
test_dataset = test_dataset.prefetch(args.batch_size)
test_dataset = iter(test_dataset)

a = next(train_datasetA)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

discA = Discriminator()
discB = Discriminator()
genA2B = Generator()
genB2A = Generator()

discA_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
discB_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
genA2B_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
genB2A_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)

if os.path.exists(args.model_save_dir) and args.start_iter != 0:
    start_iter = args.start_iter
    discA.load_weights(args.model_save_dir + '/discA_' + str(start_iter))
    discB.load_weights(args.model_save_dir + '/discB_' + str(start_iter))
    genA2B.load_weights(args.model_save_dir + '/genA2B_' + str(start_iter))
    genB2A.load_weights(args.model_save_dir + '/genB2A_' + str(start_iter))
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
        title = ['input1', 'input2', 'output1', 'output2']
    else:
        title = ['A', 'B', 'A2B', 'B2A']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        # plt.imshow(((display_list[i]+1) / 2 * 255).astype(np.uint8))
        plt.imshow((display_list[i]+1)/2)
        plt.axis('off')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.savefig(save_dir + '/generated_%d.png'%epoch)
    plt.close()


def merge_image(low_res_input, img_list):

    img_list[-1] = low_res_input
    output = pyramid.merge(img_list)

    return output

def train(train_datasetA, train_datasetB, args, lsgan=True):
    procname = os.path.basename(args.model_save_dir)
    setproctitle.setproctitle('train_{}'.format(procname))

    start = time.time()

    for iteration in range(args.start_iter, args.max_iterations):

        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
            try:
                trainA_full = next(train_datasetA)
                trainB_full = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            split_list_A = pyramid.split(trainA_full, args.level)
            split_list_B = pyramid.split(trainB_full, args.level)

            trainA = split_list_A[-1]
            trainB = split_list_B[-1]
            
            genA2B_output = genA2B(trainA, training=True)
            genB2A_output = genB2A(trainB, training=True)

            genA2B_output_full = merge_image(genA2B_output, split_list_A)
            genB2A_output_full = merge_image(genB2A_output, split_list_B)

            split_output_A = pyramid.split(genA2B_output_full, args.level)
            split_output_B = pyramid.split(genB2A_output_full, args.level)

            reconstructed_A_low = genB2A(split_output_A[-1], training=True)
            reconstructed_B_low = genA2B(split_output_B[-1], training=True)

            # reconstructed_A = genB2A(genA2B_output, training=True)
            # reconstructed_B = genA2B(genB2A_output, training=True)

            # genA2B_output = genA2B(trainA, training=True)
            # genB2A_output = genB2A(trainB, training=True)

            reconstructedA_full = merge_image(reconstructed_A_low, split_output_A)
            reconstructedB_full = merge_image(reconstructed_B_low, split_output_B)

            discA_real_output = discA(trainA_full, training=True)
            discB_real_output = discB(trainB_full, training=True)

            discA_fake_output = discA(genB2A_output_full, training=True)
            discB_fake_output = discB(genA2B_output_full, training=True)

            # Use history buffer of 50 for disc loss
            discA_loss = discriminator_loss(discA_real_output, discA_fake_output, lsgan=lsgan)
            discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=lsgan)

            generatorA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan)
            cycleA2B_loss = cycle_consistency_loss(trainA_full, trainB_full, reconstructedA_full, reconstructedB_full,
                                                 cyc_lambda=args.cyc_lambda)
            
            generatorB2A_loss = generator_loss(discA_fake_output, lsgan=lsgan)
            cycleB2A_loss = cycle_consistency_loss(trainA_full, trainB_full, reconstructedA_full, reconstructedB_full,
                                                 cyc_lambda=args.cyc_lambda)
            

            genA2B_loss = args.dis_lambda * generatorA2B_loss + cycleA2B_loss
                          
            genB2A_loss = args.dis_lambda * generatorB2A_loss + cycleB2A_loss
                          

        genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)
        genB2A_gradients = genB2A_tape.gradient(genB2A_loss, genB2A.trainable_variables)

        discA_gradients = discA_tape.gradient(discA_loss, discA.trainable_variables)
        discB_gradients = discB_tape.gradient(discB_loss, discB.trainable_variables)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))
        genB2A_optimizer.apply_gradients(zip(genB2A_gradients, genB2A.trainable_variables))

        discA_optimizer.apply_gradients(zip(discA_gradients, discA.trainable_variables))
        discB_optimizer.apply_gradients(zip(discB_gradients, discB.trainable_variables))

        if iteration % args.log_interval == 0:

            # print('Training Iteration: {}th, LOSSES: D_A: {}, D_B: {}, G_A: {}, G_B: {}, cycle: {}'.format(
            #     iteration, discA_loss, discB_loss, generatorA2B_loss, generatorB2A_loss, cycleA2B_loss))
            # print('Training Iteration: {}th, LOSSES: cycle: {:.4f}'.format(
            #     iteration, cycleA2B_loss))

            print('Training ' + procname + ', Iteration: {}th, time: {:.4f}, LOSSES: cycle: {:.4f}, discir: {:.4f}'.format(iteration, time.time() - start, cycleA2B_loss, generatorA2B_loss))

            start = time.time()

        if iteration % args.save_interval == 0:
            generate_images(trainA_full, trainB_full, genB2A_output_full, genA2B_output_full, args.save_dir_train, iteration)

            # print('Time taken for iteration {} is {} sec'.format(iteration + 1, time.time() - start))

            if not os.path.exists(args.model_save_dir):
                os.mkdir(args.model_save_dir)
            # checkpoint.save(file_prefix = checkpoint_prefix)
            discA.save_weights(args.model_save_dir + '/discA_' + str(iteration))
            discB.save_weights(args.model_save_dir + '/discB_' + str(iteration))
            genA2B.save_weights(args.model_save_dir + '/genA2B_' + str(iteration))
            genB2A.save_weights(args.model_save_dir + '/genB2A_' + str(iteration))  
        
        if iteration % args.test_interval == 0:

            input_images = []
            output_images = []

            for _ in range(args.test_size):

                test_full = next(test_dataset)
                split_list_test = pyramid.split(test_full, args.level)
                test_low = split_list_test[-1]
                generated_low = genA2B(test_low, training=False)
                generated_full = merge_image(generated_low, split_list_test)
                input_images.append(test_full)
                output_images.append(generated_full)

            generate_images(input_images[0], input_images[1], output_images[0], output_images[1], args.save_dir_test, iteration)


if __name__ == '__main__':

    train(train_datasetA, train_datasetB, args, lsgan=True)
