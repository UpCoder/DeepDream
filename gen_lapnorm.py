import tensorflow as tf
import numpy as np
from PIL import Image

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)


def lap_split(img):
    with tf.variable_scope('split'):
        '''
        使用卷积操作代替平滑操作,得到低频的部分
        '''
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')

        lo2 = tf.nn.conv2d_transpose(lo, k5x5, tf.shape(img), strides=[1, 2, 2, 1], padding='SAME')

        ho = img - lo2
    return lo, ho


def lap_split_n(img, n):
    levels = []
    for i in range(n):
        lo, ho = lap_split(img)
        levels.append(ho)
        img = lo
    levels.append(img)
    return levels[::-1]


def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.variable_scope('name'):
            # 为什么要加个hi
            img = tf.nn.conv2d_transpose(img, k5x5, tf.shape(hi), strides=[1, 2, 2, 1], padding='SAME') + hi
    return img


def normalize_std(img, eps=1e-10):
    with tf.variable_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)


def lap_normalize(img, scale_n=4):
    levels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, levels))
    out = lap_merge(tlevels)
    return out

if __name__ == '__main__':
    img_path = '/home/give/PycharmProjects/DeepLearningPractice/CH4/gen_multiscale_139/gen_multiscale200.jpg'
    img_value = np.asarray(Image.open(img_path), np.float32)
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_value, 0))

    levels = lap_split_n(img_tensor, 3)
    reconstruct = lap_merge(levels)
    reconstruct_norm = lap_normalize(img_tensor, 3)
    with tf.Session() as sess:
        lo_img, ho1_img, ho_img, reconstruct_img, reconstruct_norm_img = sess.run(
            [levels[0], levels[1], levels[-1], reconstruct, reconstruct_norm])
        # lo_img = Image.fromarray(np.asarray(np.squeeze(lo_img), np.uint8))
        # lo_img.show()
        # ho_img = Image.fromarray(np.asarray(np.squeeze(ho_img), np.uint8))
        # ho_img.show()
        reconstruct_img = Image.fromarray(np.asarray(np.squeeze(reconstruct_img), np.uint8))
        reconstruct_img.show()

        reconstruct_norm_img = Image.fromarray(np.asarray(np.squeeze(reconstruct_norm_img * 255.0), np.uint8))
        reconstruct_norm_img.show()