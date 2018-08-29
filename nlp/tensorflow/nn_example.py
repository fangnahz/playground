# coding: utf-8
import gzip
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve


url = 'http://yann.lecun.com/exdb/mnist/'
files_train = [
    ('train-images-idx3-ubyte.gz', 9912422),
    ('train-labels-idx1-ubyte.gz', 28881)
]
files_test = [
    ('t10k-images-idx3-ubyte.gz', 1648877),
    ('t10k-labels-idx1-ubyte.gz', 4542)
]


def maybe_download(url, filename, expected_bytes, force=False):
    ''' 如果文件不存在，下载并验证文件大小'''
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ignore_headers = urlretrieve(url + filename, filename)
        print('\nDownload complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception('Failed to verify ' + filename + '. Can you get it with a browser?')
    return filename

for filename, expected_bytes in (files_train + files_test):
    maybe_download(url, filename, expected_bytes)


def read_mnist(fname_img, fname_lbl):
    print('\nReading files %s and %s' % (fname_img, fname_lbl))
    with gzip.open(fname_img) as fimg:
        _ignore_magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        print(num, rows, cols)
        img = (np.frombuffer(fimg.read(num*rows*cols), dtype=np.uint8).reshape(num, rows*cols)).astype(np.float32)
        print('(Images) Returned a tensor of shape ', img.shape)
        # Standardizing the images
        img = (img - np.mean(img)) / np.std(img)
    with gzip.open(fname_lbl) as flbl:
        # flbl.read(8) reads up to 8 bytes
        _ignore_magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.frombuffer(flbl.read(num), dtype=np.int8)
        print('(Lables) Returned a tensor of shape: %s' % lbl.shape)
        print('Sample lables: ', lbl[:10])
    return img, lbl

train_inputs, train_lables = read_mnist(*[name for name, _ignore_size in files_train])
test_inputs, test_lables = read_mnist(*[name for name, _ignore_size in files_test])

# Defining hyperparameters and other constants
WEIGHTS_STRING = 'wights'
BIAS_STRING = 'bias'
batch_size = 100
img_width, img_hight = 28, 28
input_size = img_hight * img_width
num_lables = 10

# Defining inputs and outputs
tf_inputs = tf.placeholder(shape=[batch_size, input_size], dtype=tf.float32, name='inputs')
tf_lables = tf.placeholder(shape=[batch_size, num_lables], dtype=tf.float32, name='labels')


# Defining the TensorFlow variables
def define_net_parameters():
    with tf.variable_scope('layer1'):
        tf.get_variable(WEIGHTS_STRING, shape=[input_size, 500], initializer=tf.random_normal_initializer(0, 0.02))
        tf.get_variable(BIAS_STRING, shape=[500], initializer=tf.random_uniform_initializer(0, 0.01))
    with tf.variable_scope('layer2'):
        tf.get_variable(WEIGHTS_STRING, shape=[500, 250], initializer=tf.random_normal_initializer(0, 0.02))
        tf.get_variable(BIAS_STRING, shape=[250], initializer=tf.random_uniform_initializer(0, 0.01))
    with tf.variable_scope('output'):
        tf.get_variable(WEIGHTS_STRING, shape=[250, 10], initializer=tf.random_normal_initializer(0, 0.02))
        tf.get_variable(BIAS_STRING, shape=[10], initializer=tf.random_uniform_initializer(0, 0.01))


# Defining calculations in the neural network
# starting from inputs to logits
# logits are the values before applying softmax to the final output
def inference(x):
    # calculations for layer 1
    with tf.variable_scope('layer1', reuse=True):
        w = tf.get_variable(WEIGHTS_STRING)
        b = tf.get_variable(BIAS_STRING)
        tf_h1 = tf.nn.relu(tf.matmul(x, w) + b, name='hidden1')
    # calculations for layer 2
    with tf.variable_scope('layer2', reuse=True):
        w = tf.get_variable(WEIGHTS_STRING)
        b = tf.get_variable(BIAS_STRING)
        tf_h2 = tf.nn.relu(tf.matmul(tf_h1, w) + b, name='hidden2')
    # calculations for output layer
    with tf.variable_scope('output', reuse=True):
        w = tf.get_variable(WEIGHTS_STRING)
        b = tf.get_variable(BIAS_STRING)
        tf_logits = tf.nn.bias_add(tf.matmul(tf_h2, w), b, name='logits')
    return tf_logits

define_net_parameters()

# defining the loss
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=inference(tf_inputs), labels=tf_lables))
# defining the optimize function
# MomentumOptimizer gives better final accuracy and convergence than GradientDescentOptimizer
tf_loss_minimize = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.01).minimize(tf_loss)

# defining predictions
tf_predictions = tf.nn.softmax(inference(tf_inputs))

# executing the graph to get classification results
session = tf.InteractiveSession()
tf.global_variables_initializer().run()
NUM_EPOCHS = 50


def accuracy(predictions, labels):
    ''' Measure the classification accuracy of some predictions (softmax outputs)
    and lables (interger class labels)'''
    return np.sum(np.argmax(predictions, axis=1).flatten() == labels.flatten())/batch_size

test_accuracy_over_time = []
train_loss_over_time = []
for epoch in range(NUM_EPOCHS):
    train_loss = []
    # Training phase
    for step in range(train_inputs.shape[0]//batch_size):
        # Creating one-hot encoded labels with labels
        # One-hot encoding digit 3 for 10-class MNIST data set will result in
        # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        labels_one_hot = np.zeros((batch_size, num_lables), dtype=np.float32)
        labels_one_hot[np.arange(batch_size), train_lables[step*batch_size: (step+1)*batch_size]] = 1.0
        # Printing the one-hot lables
        if epoch == 0 and step == 0:
            print('Sample labels (one-hot)')
            print(labels_one_hot[:10])
            print()
        # Running the optimization process
        loss, _ = session.run([tf_loss, tf_loss_minimize], feed_dict={
            tf_inputs: train_inputs[step*batch_size: (step+1)*batch_size, :],
            tf_lables: labels_one_hot
        })
        train_loss.append(loss)  # Used to average the loss for a single epoch
    test_accuracy = []
    # Testing phase
    for step in range(test_inputs.shape[0]//batch_size):
        test_predictions = session.run(tf_predictions, feed_dict={tf_inputs: test_inputs[step*batch_size: (step+1)*batch_size, :]})
        batch_test_accuracy = accuracy(test_predictions, test_lables[step*batch_size: (step+1)*batch_size])
        test_accuracy.append(batch_test_accuracy)

    print('Averageg train loss for the %d epoch: %.3f\n' % (epoch+1, np.mean(train_loss)))
    train_loss_over_time.append(np.mean(train_loss))
    print('\tAverage test accuracy for the %d epoch: %.2f\n' % (epoch+1, np.mean(test_accuracy)*100.0))
    test_accuracy_over_time.append(np.mean(test_accuracy)*100)

session.close()

# Visualizing the loss and accuracy
x_axis = np.arange(len(train_loss_over_time))
fig, ax = plt.subplots(nrows=1, nclos=2)
fig.set_size_inches(w=25, h=5)
ax[0].plot(x_axis, train_loss_over_time)
ax[0].set_xlabel('Epochs', fontsize=18)
ax[0].set_ylable('Average train loss', fontsize=18)
ax[0].set_title('Training loss over time', fontsize=20)
ax[1].plot(x_axis, test_accuracy_over_time)
ax[1].set_xlabel('Epochs', fontsize=18)
ax[1].set_ylable('Test accuracy', fontsize=18)
ax[1].set_title('Test accuracy over time', fontsize=20)
fig.savefig('mnist_stats.jpg')
