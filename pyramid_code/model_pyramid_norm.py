import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
import scipy.stats as st
import vgg
from functools import reduce

class Residual(keras.Model):

    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.lrelu = keras.layers.LeakyReLU()

    def call(self, inputs, gamma, beta, training=True):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = adaptive_instance_norm(x, gamma, beta, epsilon=1e-5)
        x = self.lrelu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv2(x)
        x = adaptive_instance_norm(x, gamma, beta, epsilon=1e-5)       

        x = tf.add(x, inputs)

        return x


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

class MLP(keras.Model):

    def __init__(self):

        
        super(MLP, self).__init__()

        self.ch = 64

        self.flatten = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(self.ch)

        self.dense2 = keras.layers.Dense(self.ch)

        self.dense_mu = keras.layers.Dense(self.ch)
        self.dense_sigma = keras.layers.Dense(self.ch)

        self.lrelu1 = keras.layers.LeakyReLU()
        self.lrelu2 = keras.layers.LeakyReLU()

    def call(self, x):

        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.nn.relu(x)

        x = self.dense2(x)
        x = tf.nn.relu(x)

        mu = self.dense_mu(x)
        sigma = self.dense_sigma(x)

        mu = tf.reshape(mu, shape=[-1, 1, 1, self.ch])
        sigma = tf.reshape(sigma, shape=[-1, 1, 1, self.ch])

        return mu, sigma


class Generator(keras.Model):

    def __init__(self, skip=False):
        super(Generator, self).__init__()

        self.skip = skip  # TODO: Add skip

        # self.mlp = MLP()

        self.conv_first = keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.bn1 = keras.layers.BatchNormalization()
        self.res1 = Residual()
        self.res2 = Residual()
        self.res3 = Residual()
        self.res4 = Residual()
        self.res5 = Residual()
        self.res6 = Residual()
        self.res7 = Residual()
        self.res8 = Residual()
        self.res9 = Residual()
        self.bn2 = keras.layers.BatchNormalization()

        self.lrelu1 = keras.layers.LeakyReLU()
        self.lrelu2 = keras.layers.LeakyReLU()

        self.conv_last = keras.layers.Conv2D(3, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

             
    def call(self, x, y, training=True):
        # print('**********************')

        # print('maxxx1: ', x.numpy().max())

        # mu, sigma = self.mlp(y)
        mu = y.numpy().mean()
        sigma = y.numpy().std()

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        # print('maxxx2: ', x.numpy().max())

        x = self.conv_first(x)

        # print('maxxx3: ', x.numpy().max())

        # x = self.bn1(x, training=training)
        # x = tf.nn.tanh(x)

        # x = tf.nn.relu(x)

        

        x = self.res1(x, mu, sigma, training)

        # print('maxxx4: ', x.numpy().max())
        # print('maxxx4: ', x.numpy().mean())
        # print('maxxx4: ', x.numpy().std())

        x = self.res2(x, mu, sigma, training)

        # print('maxxx5: ', x.numpy().max())
        # print('**********************')

        x = self.res3(x, mu, sigma, training)
        x = self.res4(x, mu, sigma, training)
        x = self.res5(x, mu, sigma, training)
        x = self.res6(x, mu, sigma, training)
        x = self.res7(x, mu, sigma, training)
        x = self.res8(x, mu, sigma, training)
        x = self.res9(x, mu, sigma, training)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv_last(x)
        x = tf.nn.tanh(x)

        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.leaky = keras.layers.LeakyReLU(0.2)


        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()


    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.leaky(x)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.leaky(x)

        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = self.leaky(x)

        x = self.conv5(x)
        # x = tf.nn.sigmoid(x) # use_sigmoid = not lsgan
        return x
class Discriminator_multiscale(keras.Model):

    def __init__(self):
        super(Discriminator_multiscale, self).__init__()

        self.conv1_0 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky1_0 = keras.layers.LeakyReLU(0.2)
        self.conv1_1 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky1_1 = keras.layers.LeakyReLU(0.2)
        self.conv1_2 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky1_2 = keras.layers.LeakyReLU(0.2)
        self.conv1_3 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky1_3 = keras.layers.LeakyReLU(0.2)
        self.conv1_4 = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.downsample_1 = keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='same')


        self.conv2_0 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky2_0 = keras.layers.LeakyReLU(0.2)
        self.conv2_1 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky2_1 = keras.layers.LeakyReLU(0.2)
        self.conv2_2 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky2_2 = keras.layers.LeakyReLU(0.2)
        self.conv2_3 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky2_3 = keras.layers.LeakyReLU(0.2)
        self.conv2_4 = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.downsample_2 = keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='same')


        self.conv3_0 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky3_0 = keras.layers.LeakyReLU(0.2)
        self.conv3_1 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky3_1 = keras.layers.LeakyReLU(0.2)
        self.conv3_2 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky3_2 = keras.layers.LeakyReLU(0.2)
        self.conv3_3 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky3_3 = keras.layers.LeakyReLU(0.2)
        self.conv3_4 = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    def call(self, inputs):

        D_logit = []

        x = self.conv1_0(inputs)
        x = self.leaky1_0(x)
        x = self.conv1_1(x)
        x = self.leaky1_1(x)
        x = self.conv1_2(x)
        x = self.leaky1_2(x)
        x = self.conv1_3(x)
        x = self.leaky1_3(x)
        x = self.conv1_4(x)

        D_logit.append(x)

        inputs = self.downsample_1(inputs)

        x = self.conv2_0(inputs)
        x = self.leaky2_0(x)
        x = self.conv2_1(x)
        x = self.leaky2_1(x)
        x = self.conv2_2(x)
        x = self.leaky2_2(x)
        x = self.conv2_3(x)
        x = self.leaky2_3(x)
        x = self.conv2_4(x)

        D_logit.append(x)

        inputs = self.downsample_2(inputs)

        x = self.conv3_0(inputs)
        x = self.leaky3_0(x)
        x = self.conv3_1(x)
        x = self.leaky3_1(x)
        x = self.conv3_2(x)
        x = self.leaky3_2(x)
        x = self.conv3_3(x)
        x = self.leaky3_3(x)
        x = self.conv3_4(x)

        D_logit.append(x)

        return D_logit




# def down_sample(x) :
#     return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

# def discriminator(self, x_init, reuse=tf.AUTO_REUSE, scope="discriminator"):
#         D_logit = []
#         with tf.variable_scope(scope, reuse=reuse):
#             for scale in range(self.n_scale) :  # 3
#                 channel = self.ch
#                 x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) + 'conv_0')
#                 x = lrelu(x, 0.2)

#                 for i in range(1, self.n_dis): # 4
#                     x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) +'conv_' + str(i))
#                     x = lrelu(x, 0.2)

#                     channel = channel * 2

#                 x = conv(x, channels=1, kernel=1, stride=1, scope='ms_' + str(scale) + 'D_logit')
#                 D_logit.append(x)

#                 x_init = down_sample(x_init)

#             return D_logit

def discriminator_losssss(type, real, fake):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0

    for i in range(n_scale) :

        if type == 'lsgan' :
            real_loss = tf.reduce_mean(tf.compat.v1.squared_difference(real[i], 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake[i]))

        if type == 'gan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

        loss.append(real_loss + fake_loss)

    return sum(loss)

def generator_losssss(type, fake):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.compat.v1.squared_difference(fake[i], 1.0))

        if type == 'gan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

        loss.append(fake_loss)

    return sum(loss)

def gauss_kernel(kernlen=7, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(7, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def color_losses(predicted, target_image, batch_size):

    predicted_blur = blur(predicted)
    target_blur = blur(target_image)

    loss_color = tf.reduce_mean(tf.pow(target_blur - predicted_blur, 2)) / (2 * batch_size)

    return loss_color

# def _tensor_size(tensor):
#     from operator import mul
#     return reduce(mul, (d for d in tensor.get_shape()[1:]), 1)

# def content_losses(enhanced, target_image, batch_size):

#     CONTENT_LAYER = 'relu5_4'
#     vgg_dir = '../data/imagenet-vgg-verydeep-19.mat'

#     enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced))
#     target_vgg = vgg.net(vgg_dir, vgg.preprocess(target_image))

#     content_size = _tensor_size(target_vgg[CONTENT_LAYER]) * batch_size
#     loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - target_vgg[CONTENT_LAYER]) / content_size

#     return loss_content

def discriminator_loss(disc_of_real_output, disc_of_gen_output):

    # real_loss = keras.losses.mean_squared_error(disc_of_real_output, tf.ones_like(disc_of_real_output))
    # generated_loss = tf.reduce_mean(tf.square(disc_of_gen_output))

    real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_of_real_output), 
                                              logits = disc_of_real_output)
    generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_of_gen_output), 
                                                   logits = disc_of_gen_output)

    total_disc_loss = (real_loss + generated_loss) * 0.5

    return total_disc_loss


def generator_loss(disc_of_gen_output):

    # gen_loss = keras.losses.mean_squared_error(disc_of_gen_output, tf.ones_like(disc_of_gen_output))
    gen_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_of_gen_output),
                                             logits = disc_of_gen_output) 

    return gen_loss


def cycle_consistency_loss(data_A, data_B, reconstructed_data_A, reconstructed_data_B):

    loss = tf.reduce_mean(tf.abs(data_A - reconstructed_data_A) + tf.abs(data_B - reconstructed_data_B))

    return loss

# def gram_loss(input_feature, target_feature):
