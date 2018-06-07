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


tf.logging.set_verbosity(tf.logging.ERROR)

url_cat = 'http://www.anishathalye.com/media/2017/07/25/cat.jpg'
url_elephant = 'https://www.safariravenna.it/wp-content/uploads/2014/09/DSCN0920.jpg'
url_ferrari='https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_slide/public/ferrari-488-gtb-rt-2016-web-0032.jpg?itok=96_ZHnuN'

demo_epsilon = 2.0 / 255.0  # a really small perturbation
demo_lr = 1.0e-2  # 9.5e-1
demo_steps = 200
demo_target = 457  # "guacamole"



def classify(img, x, sess, labels, probs, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
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

def get_inception_v3_model(input, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(input, 0), 0.5), 2.0, name="inception_input_preprocessed")
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:] # ignore background class
        probs = tf.nn.softmax(logits, name="inception_probs") # probabilities
    return logits, probs