import tensorflow as tf
import numpy as np
import cv2
from scipy.misc import imresize
import os
flag_eliminate_border = True


# 保存图片
def savearray(img_array, img_name):
    cv2.imwrite(img_name, img_array)
    print('img saved: %s' % os.path.basename(img_name))


def calc_grads_tiled(sess, grads, input_tensor, img, tiled_size):
    '''
    只是计算grads, 注意这里的grads也是一个二维图
    :param sess:
    :param grads: 梯度tensor, 注意这里的grads和input_tensor的形状是一样的
    :param input_tensor: 输入占位符 tensor
    :param img:  输入的数据 numpy
    :param tiled_size: 分区的大小
    :return:
    '''
    shape = list(np.shape(img))
    if flag_eliminate_border:
        random_shift = np.random.random_integers(tiled_size, size=2)
        img = np.roll(np.roll(img, random_shift[0], 0), random_shift[1], 1)
    grads_value = np.zeros_like(img)
    for x in range(0, shape[0], tiled_size):
        for y in range(0, shape[1], tiled_size):
            cropped_img = img[x: x + tiled_size, y: y + tiled_size, :]
            cropped_grads_value = sess.run(grads, {
                input_tensor: cropped_img
            })
            grads_value[x: x + tiled_size, y: y + tiled_size, :] = cropped_grads_value
    if flag_eliminate_border:
        grads_value = np.roll(np.roll(grads_value, -random_shift[0], 0), -random_shift[1], 1)
    return grads_value


def gen_anysize(sess, pred_tensor, input_tensor, iter_num=20, lr=1.0):
    pred_tensor = tf.reduce_mean(pred_tensor)
    grads = tf.gradients(pred_tensor, input_tensor)[0]
    img = np.random.uniform(size=(777, 777, 3)) + 100.0
    for i in range(iter_num):
        grads_value = calc_grads_tiled(sess, grads, input_tensor, img, 128)
        grads_value /= grads_value.std() + 1e-8
        img += grads_value * lr
        # print('Iter: %d, t_score: %.4f' % (i, pred_value))
    savearray(img, './gen_multiscale' + str(iter_num) + '.jpg')


def resize_img(img, ratio):
    img = np.asarray(img, dtype=np.float32)
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255.0
    img = np.float32(imresize(img, ratio))
    img = img / 255.0 * (max - min) + min
    return img


def render_multiscale(sess, pred_tensor, input_tensor, iter_num=20, lr=1.0, octave_n=3, octave_scale=1.4):
    pred_tensor = tf.reduce_mean(pred_tensor)
    grads = tf.gradients(pred_tensor, input_tensor)[0]
    img = np.random.uniform(size=(224, 224, 3)) + 100.0
    for i in range(octave_n):
        if i > 0:
            img = resize_img(img, octave_scale)
        for j in range(iter_num):
            grads_value = calc_grads_tiled(sess, grads, input_tensor, img, 128)
            grads_value /= grads_value.std() + 1e-8
            img += grads_value * lr
    savearray(img, './gen_multiscale' + str(iter_num) + '.jpg')

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # 加载模型文件(已经包括参数啦)
    model_fn = '/home/give/PycharmProjects/DeepLearningPractice/CH4/Inception/tensorflow_inception_graph.pb'
    with tf.gfile.FastGFile(model_fn, mode='rb') as f:
        # GraphDef 就是用来加载模型的
        # 因为我们有时候需要将模型提取出来,变成与语言无关的,所以这时会就需要一个api来让我们解析模型文件
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='input-image')
    mean_pixel_value = 177.0
    processed_tf_input = tf.expand_dims(tf_input - mean_pixel_value, axis=0)
    tf.import_graph_def(graph_def, {'input': processed_tf_input})   # 将解析得到的模型加载进我们的computing graph中

    layers = [op.name for op in sess.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]

    print(layers)
    print('Number of layers: ', len(layers))
    target_name = 'mixed4d_3x3_bottleneck_pre_relu' # 可视化目标卷积层的名字
    channel = 50   #　可视化该卷积层第channel个通道
    try:
        pred_tensor = sess.graph.get_tensor_by_name('import/' + target_name + ':0')
        print('Pred_tensor: ', pred_tensor)
    except Exception as e:
        print(e)
        assert False

    pred_value = sess.run(pred_tensor, {
        tf_input: np.random.randint(1, 255, size=[512, 512, 3])
    })

    # for iter_num in [1, 10, 50, 100, 200]:
    for iter_num in [400]:
        # gen_anysize(sess, pred_tensor[:, :, :, channel], tf_input, iter_num=iter_num, lr=1.0)
        render_multiscale(sess, pred_tensor[:, :, :, channel], tf_input, iter_num=iter_num, lr=1.0)