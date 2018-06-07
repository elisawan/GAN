# https://www.anishathalye.com/2017/07/25/synthesizing-adversarial-examples/
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
import json
import PIL
import numpy as np
import pylab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from vgg16.classes import classes



tf.logging.set_verbosity(tf.logging.ERROR)

url_cat = 'http://www.anishathalye.com/media/2017/07/25/cat.jpg'
url_elephant = 'https://www.safariravenna.it/wp-content/uploads/2014/09/DSCN0920.jpg'
url_ferrari='https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_slide/public/ferrari-488-gtb-rt-2016-web-0032.jpg?itok=96_ZHnuN'

demo_epsilon = 20.0 # 2.0 / 255.0  # a really small perturbation
demo_lr = 100.0  # 9.5e-1
demo_steps = 500
demo_target = 924  # "guacamole"



def get_vgg16_model(input, reuse):
    preprocessed = tf.expand_dims(input, 0)
    arg_scope = nets.vgg.vgg_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.vgg.vgg_16(preprocessed, 1000, is_training=False)
        logits = logits[:, :] # ignore background class
        probs = tf.nn.softmax(logits, name="vgg_probs") # probabilities
    return logits, probs


# def get_inception_v3_model(input, reuse):
#     preprocessed = tf.multiply(tf.subtract(tf.expand_dims(input, 0), 0.5), 2.0, name="inception_input_preprocessed")
#     arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
#     with slim.arg_scope(arg_scope):
#         logits, _ = nets.inception.inception_v3(
#             preprocessed, 1001, is_training=False, reuse=reuse)
#         logits = logits[:, 1:] # ignore background class
#         probs = tf.nn.softmax(logits, name="inception_probs") # probabilities
#     return logits, probs


def classify(img, x, sess, labels, probs, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)


    print(img.shape)
    p = sess.run(probs, feed_dict={x: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()
    return p

def display_diff(img1, img2):

    # Calculate the absolute difference on each channel separately
    error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
    error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
    error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))

    # Calculate the maximum error for each pixel
    lum_img = np.maximum(np.maximum(error_r, error_g), error_b)

    # Uncomment the next line to turn the colors upside-down
    #lum_img = np.negative(lum_img);

    imgplot = plt.imshow(lum_img)

    # Choose a color palette
    imgplot.set_cmap('jet')
    #imgplot.set_cmap('Spectral')

    plt.colorbar()
    plt.axis('off')

    pylab.show()

# def restore_imagenet_model_from_web(sess):
#     # Make temp directory to store inception_v3.ckpt
#     data_dir = tempfile.mkdtemp()
#     inception_tarball, _ = urlretrieve('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
#     tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)
#     restore_vars = [var for var in tf.global_variables() if var.name.startswith('InceptionV3/')]
#     saver = tf.train.Saver(restore_vars)
#     saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))


def restore_vgg_model_locally(sess):
    "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
    restore_vars = [var for var in tf.global_variables() if var.name.startswith('vgg_16/')]
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, os.path.join('./vgg16/', 'vgg_16.ckpt'))

# def restore_imagenet_model_locally(sess):
#     restore_vars = [var for var in tf.global_variables() if var.name.startswith('InceptionV3/')]
#     saver = tf.train.Saver(restore_vars)
#     saver.restore(sess, os.path.join('./imagenet/', 'inception_v3.ckpt'))

def get_imagenet_labels():
    # Get imagenet labels
    return [v[0] for v in classes.values()]

def open_image(img_url):
    # Open image
    #img_path, _ = urlretrieve(img_url)
    img_path = img_url #urlretrieve(img_url)

    # img = PIL.Image.open(img_path)
    #
    # # resize image
    # big_dim = max(img.width, img.height)
    # wide = img.width > img.height
    # new_w = 224 if not wide else int(img.width * 224 / img.height)
    # new_h = 224 if wide else int(img.height * 224 / img.width)
    # img = img.resize((new_w, new_h)).crop((0, 0, 224, 224))
    # img = (np.asarray(img)).astype(np.float32)

    im_original = cv2.resize(cv2.imread(img_url), (224, 224))
    im = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
    #im = im.transpose((2, 1, 0))
    im_converted = np.expand_dims(im, axis=0)
    return im

def main():


    # Open tf session
    sess = tf.InteractiveSession()

    # Prepare tf variable for an image
    image = tf.Variable(tf.zeros((224, 224, 3)), name="variable_input")

    logits, probs = get_vgg16_model(image, reuse=False)

    restore_vgg_model_locally(sess)
    imagenet_labels = get_imagenet_labels()

    img = open_image("/Users/francescoventura/PycharmProjects/ebano-webapp/static/uploads/africanelephant.png")
    img_class = None
    print(np.shape(img))

    plt.imshow(img)
    plt.show()
    plt.close()



    x_hat = image # trainable adversarial input
    x = tf.placeholder(tf.float32, (224, 224, 3), name='x')
    assign_op = tf.assign(x_hat, x)

    # target
    y_hat = tf.placeholder(tf.int32, (), name="y_hat")
    labels = tf.one_hot(y_hat, 1000, name="labels")

    # hyper parameters
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    epsilon = tf.placeholder(tf.float32, (), name="epsilon")

    below = x - epsilon
    above = x + epsilon

    # clipping between 0 and 1
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above, name="clip-epsilon"), 0, 255, name="clip-0-1")

    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    # loss function
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])  # + tf.nn.l2_loss(x_hat-x_old, name='loss')

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

    # initialization step
    sess.run(assign_op, feed_dict={x: img})


    # projected gradient descent
    for i in range(demo_steps):
        # gradient descent step
        _, loss_value = sess.run(
            [optimizer, loss],
            feed_dict={learning_rate: demo_lr, y_hat: demo_target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i + 1) % 10 == 0:
            print('step %d, loss=%g' % (i + 1, loss_value))

        if (i + 1) % 50 == 0:
            adv = x_hat.eval()
            #print(adv)
            plt.imshow(adv.astype(int))
            plt.show()
            plt.close()

    adv = x_hat.eval()  # retrieve the adversarial example
    print(adv.astype(int))
    p_o =classify(img, image, sess, imagenet_labels, probs, correct_class=img_class)
    p_m =classify(adv.astype(int), image, sess, imagenet_labels, probs, correct_class=img_class, target_class=demo_target)
    display_diff(adv.astype(int), img)

    # Set the logs writer to the folder ./logs

    train_writer = tf.summary.FileWriter('./logs/vgg16/',
                                         graph=sess.graph)
    train_writer.close()



if __name__ == '__main__':
    main()