# _*_ coding:utf-8 _*_

'''
This is a Conditional Generative Adversarial Networks test.
date: 2018-8-27
Author: MrRenQIANG
'''

import tensorflow as tf
import numpy as np
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

path_root =r'C:\Users\admin\PycharmProjects\py_projects\tensorflow_learnining\gan_test1\generative-models'
dataset = r'MNIST_data'
path = os.path.join(path_root, dataset)
mnist = input_data.read_data_sets(path, one_hot=True)

batch_size = 64
Z_dim = 100
X_dim = 784
y_dim = 10
h_dim = 128
lr = 0.001

# W random_normal initialization
def w_init(size):
    in_dim = size[0]
    n_stddev = 1./tf.sqrt(in_dim/2.0)
    return tf.random_normal(shape=size, stddev=n_stddev)


# The Discrimination Net model
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_w1 = tf.Variable(w_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_w2 = tf.Variable(w_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

vars_D = [D_w1, D_w2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(values=[x, y], axis=1)
    D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, D_w1) + D_b1)
    D_logits = tf.matmul(D_h1, D_w2) + D_b2
    D_prob = tf.nn.sigmoid(D_logits)
    return  D_prob, D_logits


# The Generative Net Model
# y can simply concat with z........can concat with z' which size is 784  ??????
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_w1 = tf.Variable(w_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_w2 = tf.Variable(w_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

vars_G = [G_w1, G_w2, G_b1, G_b2]


def generative(z, y):
    #  input must contain y? or y should concat on generative but inputs
    inputs = tf.concat(values=[z, y], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_w1) + G_b1)
    G_logits = tf.matmul(G_h1, G_w2) + G_b2
    G_prob = tf.nn.sigmoid(G_logits)
    return G_prob


# z--uniform distribute
def z_sample(m, n):
    """
    :param m: z batch
    :param n: z dimension
    :return:  z input samples
    """
    return np.random.uniform(-1, 1, size=[m, n])

def plot_samples(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        axes = plt.subplot(gs[i])
        plt.axis("off")
        axes.set_xticklabels([])  # set x label font
        axes.set_yticklabels([])
        axes.set_aspect('equal')  # length of x,y is equal
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_samples = generative(Z, y)
D_real, D_logits = discriminator(X, y)
D_fake, D_log_fake = discriminator(G_samples, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_log_fake, labels=tf.zeros_like(D_log_fake)))

with tf.name_scope('d_loss'):
    D_loss = D_loss_real + D_loss_fake
    tf.summary.scalar('d_loss', D_loss)
with tf.name_scope('g_loss'):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_log_fake, labels=tf.ones_like(D_log_fake)))
    tf.summary.scalar('g_loss', G_loss)


# global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.01, global_step=global_step, decay_steps=1000, decay_rate=0.9)
D_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=vars_D)
G_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=vars_G)


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(r'./log2', tf.get_default_graph())

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out2/'):
    os.makedirs('out2/')

i = 0
for n in range(1000000):
    if n % 1000 == 0:
        n_samples = 25

        Z_sample = z_sample(n_samples, Z_dim)
        y_samples = np.zeros(shape=[n_samples, y_dim])
        y_samples[:, 5] = 1

        sampples = sess.run(G_samples, feed_dict={Z:Z_sample, y:y_samples})

        fig = plot_samples(sampples)
        plt.savefig('out2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_train, y_labels = mnist.train.next_batch(batch_size)

    Z_sample = z_sample(batch_size, Z_dim)
    _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_train, Z: Z_sample, y: y_labels})
    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: Z_sample, y: y_labels})

    result = sess.run(merged, feed_dict={X: X_train, Z: Z_sample, y: y_labels})
    writer.add_summary(result, n)

    if n % 1000 == 0:
        print('Iter: {}'.format(n))
        print('D loss: ', D_loss_curr)
        print('G_loss: ', G_loss_curr)
        print()
