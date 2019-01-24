from datetime import datetime
import sys
import os
import tensorflow as tf
import math
import numpy.linalg as la
import numpy as np
import scipy.io
from dataset import Dataset

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    weight_file = sys.argv[3]

    # necessary params
    v_factor = 1.7
    regularization_factor = 0.1
    learning_rate = 0.001
    training_iters = 144000
    save_step = 28800
    batch_size = 150
    display_step = 20
    test_step = 20
    r = 2
    M= 59
    N= 283
    dropout_keep_rate = 0.5
    image_size = 200
    image_channel = 3
    dx_factor = 1.5
    dD_factor = 1.5

    # Graph input
    input_img = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel])
    ygen_gt = tf.placeholder(tf.float32, [batch_size, M*r])
    xhat_gt = tf.placeholder(tf.float32, [batch_size, N*r])

    # ResNet prediction
    # ResNet architecture to be used: 50, 101 or 152
    # call the ResNetModel class
    model = ResNetModel(is_training: True, depth=50, num_classes=M*r)
    u_predict = model.inference(input_img)
    u_predict_t = tf.transpose(u_predict)
    u_predict_t = tf.reshape(u_predict,[M,r*batch_size])

    # Now build LISTA
    initial_theta = 0.1
    T = 100
    G = scipy.io.loadmat('base-matrix-200-287.mat')
    G = G['G']
    G = G.astype(np.float32)

    We = tf.Variable(tf.transpose(G),name='We')
    S = tf.Variable(np.identity(N) - tf.matmul(tf.transpose(G),G), name='S')
    theta = tf.Variable(initial_theta,name='theta')

    def shrinkage(Z,theta):
        Z = tf.sign(Z) * tf.maximum(tf.abs(Z) - theta, 0)
        return Z

    B = tf.matmul(We,u_predict_t)
    Z = shrinkage(B,theta)
    def body(i,Znew):
        Znew = shrinkage(B + tf.matmul(S,Znew),theta)
        i = i + 1
        return i,Znew
    def cond(i,*args):
        return i<T
    k,Z = tf.while_loop(cond,body,[0,Z])
    Z = tf.transpose(Z)
    Z = tf.reshape(Z,[batch_size,N*r])
    #print(Z,file=sys.stderr)

    # Backpropagation with the proposed rule
    # delta-x part
    dZ = tf.sign(Z - xhat_gt)
    dZ = tf.reshape(dZ,[batch_size*r,N])
    Z = tf.reshape(Z,[batch_size*r,N])
    Map = tf.cast(tf.greater_equal(tf.abs(Z),1e-8),tf.float32)
    Z = tf.reshape(Z,[batch_size,N*r])
    du_predict = tf.matmul(Map * dZ, tf.transpose(G))
    du_predict = tf.stop_gradient(du_predict)
    du_predict = tf.reshape(du_predict,[batch_size,M*r])
    dx_value = tf.reduce_mean(u_predict * du_predict)
    
    #delta-D part
    Z_t = tf.reshape(Z, [batch_size*r,N])
    temp = tf.matmul(Z_t, We)
    temp = tf.reshape(temp, [batch_size,M*r])
    temp = u_predict - temp 
    temp = tf.reshape(temp,[batch_size*r,M])
    temp = tf.matmul(tf.transpose(Map), temp)
    dWe = temp
    temp = tf.matmul(Map * dZ, We)
    temp = tf.matmul(tf.transpose(Z_t), temp)
    dWe = dWe - temp
    dD_value = tf.reduce_mean(We * dWe)
    dWe = tf.stop_gradient(dWe)

    # Loss and optimizer
    # now we need gt output of CNN, and gt output of LISTA, both of them are read from disk
    loss = tf.reduce_mean(tf.squared_difference(u_predict, ygen_gt)) + dx_factor * dx_value + dD_factor * dD_value

    loss_CNN = tf.reduce_mean(tf.squared_difference(u_predict, ygen_gt))
    loss_LISTA = tf.reduce_mean(tf.abs(Z - xhat_gt))

    # l2 regularization for parameters in CNN and in LISTA
    regularizers = (tf.nn.l2_loss(conv1_weights) +
                tf.nn.l2_loss(conv2_weights) +
                tf.nn.l2_loss(conv3_weights) +
                tf.nn.l2_loss(conv4_weights) +
                tf.nn.l2_loss(conv5_weights) +
                tf.nn.l2_loss(fc1_weights) +
                tf.nn.l2_loss(fc2_weights) +
                tf.nn.l2_loss(fc3_weights) +
                tf.nn.l2_loss(We) + tf.nn.l2_loss(S))

    loss = loss + regularization_factor / batch_size * regularizers

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Init
    init = tf.global_variables_initializer()

    # Add ops for saving all the variables later.
    saver = tf.train.Saver()

    # Load dataset
    dataset = Dataset(train_file, test_file)

    # Launch the graph

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    #with tf.Session() as sess:
        print('Init variable')
        sess.run(init)
        
        print('Start training')
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            #split the batch_ys to ygen_gt and xhat_gt
            vg = batch_ys[:,0:N*r]
            ug = batch_ys[:,N*r:N*r+M*r]
            #print(ug, file=sys.stderr)
            sess.run(optimizer, feed_dict={input_img: batch_xs, ygen_gt: ug, xhat_gt: vg})

            # Display training status
            if step % display_step == 0:
                batch_loss = sess.run(loss, feed_dict={input_img: batch_xs, ygen_gt: ug, xhat_gt: vg})
                print('{} Iter {}: Total loss = {:.4f}'.format(datetime.now(), step, batch_loss), file=sys.stderr)
                print('{} Iter {}: Total loss = {:.4f}'.format(datetime.now(), step, batch_loss))

                batch_loss_CNN = sess.run(loss_CNN, feed_dict={input_img: batch_xs, ygen_gt: ug})
                print('{} Iter {}: CNN loss = {:.4f}'.format(datetime.now(), step, batch_loss_CNN), file=sys.stderr)
                print('{} Iter {}: CNN loss = {:.4f}'.format(datetime.now(), step, batch_loss_CNN))

                batch_loss_LISTA = sess.run(loss_LISTA, feed_dict={input_img: batch_xs, xhat_gt: vg})
                print('{} Iter {}: LISTA loss = {:.4f}'.format(datetime.now(), step, batch_loss_LISTA), file=sys.stderr)
                print('{} Iter {}: LISTA loss = {:.4f}'.format(datetime.now(), step, batch_loss_LISTA))
	
            step += 1

        ## Save model to disk after every save_step iterations.
        # /path-of-model-saved-to
        saver.save(sess, '/trained-models/ECNNCS-',global_step=save_step,write_meta_graph=False)
        print ('Model saved', file=sys.stderr)
        print('Finish!')



class ResNetModel(object):

    def __init__(self, is_training, depth=50, num_classes=1000):
        self.is_training = is_training
        self.num_classes = num_classes
        self.depth = depth

        if depth in NUM_BLOCKS:
            self.num_blocks = NUM_BLOCKS[depth]
        else:
            raise ValueError('Depth is not supported; it must be 50, 101 or 152')


    def inference(self, x):
        # Scale 1
        with tf.variable_scope('scale1'):
            s1_conv = conv(x, ksize=7, stride=2, filters_out=64)
            s1_bn = bn(s1_conv, is_training=self.is_training)
            s1 = tf.nn.relu(s1_bn)

        # Scale 2
        with tf.variable_scope('scale2'):
            s2_mp = tf.nn.max_pool(s1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2 = stack(s2_mp, is_training=self.is_training, num_blocks=self.num_blocks[0], stack_stride=1, block_filters_internal=64)

        # Scale 3
        with tf.variable_scope('scale3'):
            s3 = stack(s2, is_training=self.is_training, num_blocks=self.num_blocks[1], stack_stride=2, block_filters_internal=128)

        # Scale 4
        with tf.variable_scope('scale4'):
            s4 = stack(s3, is_training=self.is_training, num_blocks=self.num_blocks[2], stack_stride=2, block_filters_internal=256)

        # Scale 5
        with tf.variable_scope('scale5'):
            s5 = stack(s4, is_training=self.is_training, num_blocks=self.num_blocks[3], stack_stride=2, block_filters_internal=512)

        # post-net
        avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')

        with tf.variable_scope('fc'):
            self.prob = fc(avg_pool, num_units_out=self.num_classes)

        return self.prob

"""
Class helper methods
"""
def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)

def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride)
    return x


def block(x, is_training, block_filters_internal, block_stride):
    filters_in = x.get_shape()[-1]

    m = 4
    filters_out = m * block_filters_internal
    shortcut = x

    with tf.variable_scope('a'):
        a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
        a_bn = bn(a_conv, is_training)
        a = tf.nn.relu(a_bn)

    with tf.variable_scope('b'):
        b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal)
        b_bn = bn(b_conv, is_training)
        b = tf.nn.relu(b_bn)

    with tf.variable_scope('c'):
        c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out)
        c = bn(c_conv, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out)
            shortcut = bn(shortcut_conv, is_training)

    return tf.nn.relu(c + shortcut)


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    return tf.nn.xw_plus_b(x, weights, biases)

def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv


if __name__ == '__main__':
    main()
