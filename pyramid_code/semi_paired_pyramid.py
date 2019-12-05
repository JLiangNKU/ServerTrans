import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pyramid

from model_pyramid import Generator, Discriminator, cycle_consistency_loss, generator_loss, discriminator_loss

# tf.random.set_seed(22)
# np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

parser = argparse.ArgumentParser()

req_grp = parser.add_argument_group('required')
req_grp.add_argument('--model_save_dir', default='../checkpoint_enhance_paired', type=str, help='directory to save checkpoints to.')
req_grp.add_argument('--task', default='enhance', type=str, help='task to do')
req_grp.add_argument('--save_dir_train', default='../sample_train_paied', type=str, help=' ')
req_grp.add_argument('--save_dir_test', default='../sample_test_paied', type=str, help=' ')
req_grp.add_argument('--start_iter', default=0, type=int, help='iterations to start training')

train_grp = parser.add_argument_group('training')
train_grp.add_argument('--learning_rate', default=2e-4, type=float, help='learning rate for the stochastic gradient update.')
train_grp.add_argument('--gan_lambda', default=10, type=float, help='weight of the gan loss.')
train_grp.add_argument('--mse_lambda', default=1, type=float, help='weight of the mse loss.')
train_grp.add_argument('--level', default=4, type=int, help='number of levels of the pyramid')
train_grp.add_argument('--max_iterations', default=100000, type=int, help='training epoches')
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
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""### Load Datasets"""

if args.task == 'enhance':

    PATH = '../data/fivek_1080p'
    train_path = os.path.join(PATH, "concated")
    test_path = os.path.join(PATH, 'test')
    print('*****trained on the image enhancement task*****')

train_size = len(os.listdir(train_path))
print('train size:', train_size)

def load_image(image_file, is_train=True):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if not is_train:
        image = tf.image.resize(image, [args.load_img_size, args.load_img_size])
    # image = (image / 127.5) - 1
    image = image * 2 - 1

    return image

train_dataset = tf.data.Dataset.list_files(train_path + '/*.jpg', shuffle=True)
train_dataset = train_dataset.shuffle(train_size).repeat(args.max_iterations)
train_dataset = train_dataset.map(lambda x: load_image(x))
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.prefetch(args.batch_size)
train_dataset = iter(train_dataset)

test_dataset = tf.data.Dataset.list_files(test_path + '/*.jpg', shuffle=True)
test_dataset = test_dataset.shuffle(500).repeat(args.max_iterations)
test_dataset = test_dataset.map(lambda x: load_image(x, is_train=False))
test_dataset = test_dataset.batch(args.batch_size)
test_dataset = test_dataset.prefetch(args.batch_size)
test_dataset = iter(test_dataset)

a = next(train_dataset)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

discB = Discriminator()
genA2B = Generator()

discB_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)
genA2B_optimizer = keras.optimizers.Adam(args.learning_rate, beta_1=0.5)

if os.path.exists(args.model_save_dir) and args.start_iter != 0:
    start_iter = args.start_iter
    discB.load_weights(args.model_save_dir + '/discB_' + str(start_iter))
    genA2B.load_weights(args.model_save_dir + '/genA2B_' + str(start_iter))
    print("Restored from checkpoints {}".format(start_iter))
else:
    print("Initializing from scratch.")

def generate_images(A, B, B2A, A2B, save_dir, epoch):

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
        plt.imshow((display_list[i] * 0.5 + 0.5))
        plt.axis('off')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.savefig(save_dir + '/generated_%d.png'%epoch)
    plt.close()


def merge_image(low_res_input, img_list):

    img_list[-1] = low_res_input
    output = pyramid.merge(img_list)

    return output

def train(train_dataset, args, lsgan=True):

    for iteration in range(args.start_iter, args.max_iterations):
        start = time.time()

        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as discB_tape:
            try:
                train_full = next(train_dataset)
                trainA_full = train_full[:, :, :1024, :]
                trainB_full = train_full[:, :, 1024:, :]

            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            split_list_A = pyramid.split(trainA_full, args.level)

            trainA = split_list_A[-1]
            
            genA2B_output = genA2B(trainA, training=True)

            genA2B_output_full = merge_image(genA2B_output, split_list_A)

            discB_real_output = discB(trainB_full, training=True)

            discB_fake_output = discB(genA2B_output_full, training=True)

            # Use history buffer of 50 for disc loss
            discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=lsgan)

            generatorA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan)

            mse_loss = tf.reduce_mean(tf.square(trainB_full - genA2B_output_full))

            genA2B_loss = args.gan_lambda * generatorA2B_loss + args.mse_lambda * mse_loss
                          

        genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)

        discB_gradients = discB_tape.gradient(discB_loss, discB.trainable_variables)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))

        discB_optimizer.apply_gradients(zip(discB_gradients, discB.trainable_variables))

        if iteration % args.log_interval == 0:

            # print('Training Iteration: {}th, LOSSES: D_A: {}, D_B: {}, G_A: {}, G_B: {}, cycle: {}'.format(
            #     iteration, discA_loss, discB_loss, generatorA2B_loss, generatorB2A_loss, cycleA2B_loss))
            print('Training Iteration: {}th, LOSSES: mse: {:.4f}'.format(
                iteration, mse_loss))

        if iteration % args.save_interval == 0:
            generate_images(trainA_full, trainB_full, genA2B_output_full, genA2B_output_full, args.save_dir_train, iteration)

            # print('Time taken for iteration {} is {} sec'.format(iteration + 1, time.time() - start))

            if not os.path.exists(args.model_save_dir):
                os.mkdir(args.model_save_dir)
            discB.save_weights(args.model_save_dir + '/discB_' + str(iteration))
            genA2B.save_weights(args.model_save_dir + '/genA2B_' + str(iteration))
        
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

            generate_images(input_images[0], input_images[1], output_images[1], output_images[0], args.save_dir_test, iteration)


if __name__ == '__main__':

    train(train_dataset, args, lsgan=True)
