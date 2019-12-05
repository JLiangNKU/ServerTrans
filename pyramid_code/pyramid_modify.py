from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import shape

def _downsample(image, kernel):
  # image = tf.pad(image, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
  # return tf.nn.conv2d(
  #     input=image, filters=kernel, strides=[1, 2, 2, 1], padding="SAME")
  return tf.nn.conv2d(
      input=image, filters=kernel, strides=[1, 2, 2, 1], padding="SAME")

def _binomial_kernel(num_channels, dtype=tf.float32):
  kernel = np.array((1., 4., 6., 4., 1.), dtype=dtype.as_numpy_dtype())
  kernel = np.outer(kernel, kernel)
  kernel /= np.sum(kernel)
  kernel = kernel[:, :, np.newaxis, np.newaxis]
  constentt = tf.constant(kernel, dtype=dtype)
  eyee = tf.eye(num_channels, dtype=dtype)
  result = constentt * eyee
  return  result


def _build_pyramid(image, sampler, num_levels):
  kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
  levels = [image]
  for _ in range(num_levels):
    image = sampler(image, kernel)
    levels.append(image)
  return levels
  
def _split(image, kernel):
  low = _downsample(image, kernel)
  high = image - _upsample(low, kernel, tf.shape(input=image))
  return high, low


def _upsample(image, kernel, output_shape=None):
  if output_shape is None:
    output_shape = tf.shape(input=image)
    output_shape = (output_shape[0], output_shape[1] * 2, output_shape[2] * 2,
                    output_shape[3])
  # image = tf.pad(image, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
  return tf.nn.conv2d_transpose(
      image,
      kernel * 4.0,
      output_shape=output_shape,
      strides=[1, 2, 2, 1],
      padding="SAME")

def downsample(image, num_levels, name=None):
  with tf.compat.v1.name_scope(name, "pyramid_downsample", [image]):
    image = tf.convert_to_tensor(value=image)

    # shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    return _build_pyramid(image, _downsample, num_levels)


def merge(levels, name=None):
  with tf.compat.v1.name_scope(name, "pyramid_merge", levels):
    levels = [tf.convert_to_tensor(value=level) for level in levels]

    # for index, level in enumerate(levels):
    #   shape.check_static(
    #       tensor=level, tensor_name="level {}".format(index), has_rank=4)

    image = levels[-1]
    kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
    for level in reversed(levels[:-1]):
      image = _upsample(image, kernel, tf.shape(input=level)) + level
    return image


def split(image, num_levels, name=None):
  with tf.compat.v1.name_scope(name, "pyramid_split", [image]):
    image = tf.convert_to_tensor(value=image)

    # shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
    low = image
    lows = []
    highs = []
    for _ in range(num_levels):
      high, low = _split(low, kernel)
      highs.append(high)
      lows.append(low)
    return highs, lows


def upsample(image, num_levels, name=None):
  with tf.compat.v1.name_scope(name, "pyramid_upsample", [image]):
    image = tf.convert_to_tensor(value=image)

    # shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    return _build_pyramid(image, _upsample, num_levels)
