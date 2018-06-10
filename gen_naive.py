import tensorflow as tf
import numpy as np
import cv2
import os


# 保存图片
def savearray(img_array, img_name):
    cv2.imwrite(img_name, img_array)
    print('img saved: %s' % os.path.basename(img_name))


def gen_naive(sess, pred_tensor, input_tensor, iter_num=20, lr=1.0):
    pred_tensor = tf.reduce_mean(pred_tensor)
    grads = tf.gradients(pred_tensor, input_tensor)[0]
    img = np.random.uniform(size=(224, 224, 3)) + 100.0
    for i in range(iter_num):
        grads_value, pred_value = sess.run([grads, pred_tensor], {
            input_tensor: img
        })
        grads_value /= grads_value.std() + 1e-8
        img += grads_value * lr
        print('Iter: %d, t_score: %.4f' % (i, pred_value))
    savearray(img, './gen_naive' + str(iter_num) + '.jpg')

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
    # target_name = 'mixed4d_3x3_bottleneck_pre_relu' # 可视化目标卷积层的名字
    # channel = 139   #　可视化该卷积层第channel个通道
    target_name = 'mixed4c'
    try:
        pred_tensor = sess.graph.get_tensor_by_name('import/' + target_name + ':0')
        print('Pred_tensor: ', pred_tensor)
    except Exception as e:
        print(e)
        assert False

    pred_value = sess.run(pred_tensor, {
        tf_input: np.random.randint(1, 255, size=[512, 512, 3])
    })

    for iter_num in [1, 10, 50, 100, 200]:
        gen_naive(sess, tf.square(pred_tensor), tf_input, iter_num=iter_num, lr=1.0)