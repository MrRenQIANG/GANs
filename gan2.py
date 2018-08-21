# _*_ coding:utf-8 _*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns  # for pretty plots
from scipy.stats import norm

# seed = 20
# np.random.seed(seed)
# tf.set_random_seed(seed)  # 设置的seed()值仅一次有效

mu, sigma = -1, 1
# xs = np.linspace(-5, 5, 1000)
# plt.plot(xs,norm.pdf(xs, loc=mu, scale=sigma))
# plt.show() # 有了%matplotlib inline 就可以省掉plt.show()了

TRAIN_STEPS = 10000
MINIBATCH_SIZE  = 200

def mlp(input, output_dim):
    w1 = tf.get_variable('w1', [input.get_shape()[1],6], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('b1',[6], initializer=tf.constant_initializer(0.0))
    w2 = tf.get_variable('w2', [6, 5], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('b2', [5], initializer=tf.constant_initializer(0.0))
    w3 = tf.get_variable('w3', [5, output_dim], initializer=tf.random_normal_initializer())
    b3 = tf.get_variable('b3', [output_dim], initializer=tf.constant_initializer(0.0))

    # nn operators
    fc1 = tf.nn.tanh(tf.matmul(input, w1) + b1)
    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
    fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)
    return fc3, [w1, b1, w2, b2, w3, b3]

def momentum_optimizer(loss, var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001,
        batch,
        TRAIN_STEPS//4,
        0.95,
        staircase=True
    )
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.6).minimize(loss, global_step=batch, var_list=var_list)
    return  optimizer


# use MSE doing d_pre
with tf.variable_scope("D_pre"):
    input_node = tf.placeholder(tf.float32, shape=[MINIBATCH_SIZE, 1])
    train_labels = tf.placeholder(tf.float32, shape=[MINIBATCH_SIZE, 1])
    D, theta = mlp(input_node, 1)
    loss = tf.reduce_mean(tf.square(D-train_labels))

optimizer = momentum_optimizer(loss, None)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# plot decision surface
# def plot_d0(D, input_node):
#     f, ax = plt.subplots(1)
#     xs = np.linspace(-5, 5, 1000)
#     ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')
#
#     r = 1000
#     xs = np.linspace(-5, 5, r)
#     ds = np.zeros((r, 1))
#     # process multiple points in parallel in a minibatch
#     for i in range(r//MINIBATCH_SIZE):
#         # 输入是一个数，200个数一起算作一个小批次
#         x = np.reshape(xs[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1)], (MINIBATCH_SIZE,1))
#         ds[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1)] = sess.run(D, {input_node: x})
#     ax.plot(xs, ds, label='decision boundary')
#     ax.set_ylim(-0.1, 1.1)
#     plt.legend()
#
# # plot_d0(D, input_node)
# # plt.title('Initial Decision Boundary')
#
# lh = np.zeros(1000)
# for i in range(1000):
#     d = (np.random.random(MINIBATCH_SIZE) - 0.5) * 10.0
#     labels = norm.pdf(d, loc=mu, scale=sigma)
#     lh[i], _ = sess.run([loss, optimizer], {input_node: np.reshape(d, (MINIBATCH_SIZE, 1)), train_labels: np.reshape(labels, (MINIBATCH_SIZE, 1))})

# plt.plot(lh)
# plt.title('TAraining Loss')

# plot_d0(D, input_node)
# plt.show()

# copy the learned weights over into a tmp array
weightsD = sess.run(theta)
# close the pre-training session
sess.close()

# build Net
with tf.variable_scope("G"):
    z_node = tf.placeholder(tf.float32, shape=(MINIBATCH_SIZE, 1))
    G, theta_g = mlp(z_node, 1)
    G = tf.multiply(5.0, G) # scale up

with tf.variable_scope("D") as scope:
    x_node = tf.placeholder(tf.float32, shape=(MINIBATCH_SIZE, 1))
    fc, theta_d = mlp(x_node, 1)
    D1 = tf.maximum(tf.minimum(fc, 0.99), 0.01)  # clamp as a probability
    # make a copy of D that uses the same variables, but takes in G as input
    scope.reuse_variables()
    fc, theta_d = mlp(G, 1)
    D2 = tf.maximum(tf.minimum(fc, 0.99), 0.01)
obj_d = tf.reduce_mean(-tf.log(D1) -  tf.log(1-D2))
obj_g = tf.reduce_mean(tf.log(1-D2))

# set up optimizer for D ,G
opt_d = momentum_optimizer(obj_d, theta_d)
opt_g = momentum_optimizer(obj_g, theta_g)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# copy weights from pre-training over to new D network
# for i, v in enumerate(theta_d):
#     sess.run(v.assign(weightsD[i]))
#
#
def plot_fig():
    f, ax = plt.subplots(1)
    # p_data
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')

    # decision boundary
    r = 5000
    xs = np.linspace(-5, 5, r)
    ds = np.zeros((r, 1))
    for i in range(r//MINIBATCH_SIZE):
        x = np.reshape(xs[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1)], (MINIBATCH_SIZE, 1))
        ds[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1)] = sess.run(D1, {x_node: x})
    ax.plot(xs, ds, label='decidion boundary')

    # distribution of inverse-mapped points
    zs = np.linspace(-5, 5, r)
    gs = np.zeros((r, 1))
    for i in range(r//MINIBATCH_SIZE):
        z = np.reshape(zs[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1)], (MINIBATCH_SIZE, 1))
        gs[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1)] = sess.run(G, {z_node: z})
    histc, edges = np.histogram(gs, bins=10)
    ax.plot(np.linspace(-5, 5, 10), histc/float(r), label='p_g')
    plt.ylim(-0.1, 1.1)
    plt.legend()
# plot_fig()
# plt.title('Before Training')
# plt.show()

# Algorithm 1 of Goodfellow et al 2014# Algori
k = 1
histd, histg = np.zeros(TRAIN_STEPS), np.zeros(TRAIN_STEPS)
for i in range(TRAIN_STEPS):
    for j in range(k):
        x = np.random.normal(mu, sigma, MINIBATCH_SIZE)
        x.sort()
        z = np.linspace(-5.0, 5.0 ,MINIBATCH_SIZE) + np.random.random(MINIBATCH_SIZE) * 0.01
        histd[i], _ = sess.run([obj_d, opt_d], {x_node: np.reshape(x, (MINIBATCH_SIZE, 1)), \
                                                z_node: np.reshape(z, (MINIBATCH_SIZE, 1))})

    z = np.linspace(-5.0, 5.0 , MINIBATCH_SIZE) + np.random.random(MINIBATCH_SIZE) * 0.01
    histg[i], _ = sess.run([obj_g, opt_g], {z_node: np.reshape(z, (MINIBATCH_SIZE, 1))})
    if i % (TRAIN_STEPS//10) == 0:
        print(float(i)/float(TRAIN_STEPS))

plt.plot(range(TRAIN_STEPS), histd, label='obj_d')
plt.plot(range(TRAIN_STEPS), 1 - histg, label='obj_g')
plt.legend()
plot_fig()
plt.show()
