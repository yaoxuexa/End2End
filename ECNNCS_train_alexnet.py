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

    # conv layer 1
    conv1_weights = tf.Variable(tf.random_normal([11, 11, image_channel, 96], dtype=tf.float32, stddev=0.01),name='conv1_weights')
    conv1_biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),name='conv1_biases')
    conv1 = tf.nn.conv2d(input_img, conv1_weights, [1, 4, 4, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, conv1_biases)
    conv1_relu = tf.nn.relu(conv1)
    conv1_norm = tf.nn.local_response_normalization(conv1_relu, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)
    conv1_pool = tf.nn.max_pool(conv1_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv layer 2
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.01),name='conv2_weights')
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),name='conv2_biases')
    conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, conv2_biases)
    conv2_relu = tf.nn.relu(conv2)
    conv2_norm = tf.nn.local_response_normalization(conv2_relu)
    conv2_pool = tf.nn.max_pool(conv2_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv layer 3
    conv3_weights = tf.Variable(tf.random_normal([3, 3, 256, 384], dtype=tf.float32, stddev=0.01),name='conv3_weights')
    conv3_biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),name='conv3_biases')
    conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, [1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, conv3_biases)
    conv3_relu = tf.nn.relu(conv3)

    # conv layer 4
    conv4_weights = tf.Variable(tf.random_normal([3, 3, 384, 384], dtype=tf.float32, stddev=0.01),name='conv4_weights')
    conv4_biases = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32),name='conv4_biases')
    conv4 = tf.nn.conv2d(conv3_relu, conv4_weights, [1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, conv4_biases)
    conv4_relu = tf.nn.relu(conv4)

    # conv layer 5
    conv5_weights = tf.Variable(tf.random_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.01),name='conv5_weights')
    conv5_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),name='conv5_biases')
    conv5 = tf.nn.conv2d(conv4_relu, conv5_weights, [1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.bias_add(conv5, conv5_biases)
    conv5_relu = tf.nn.relu(conv5)
    conv5_pool = tf.nn.max_pool(conv5_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc layer 1
    fc1_weights = tf.Variable(tf.random_normal([256 * 5 * 5, 4096], dtype=tf.float32, stddev=0.01),name='fc1_weights')
    fc1_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),name='fc1_biases')
    conv5_reshape = tf.reshape(conv5_pool, [-1, fc1_weights.get_shape().as_list()[0]])
    fc1 = tf.matmul(conv5_reshape, fc1_weights)
    fc1 = tf.nn.bias_add(fc1, fc1_biases)
    fc1_relu = tf.nn.relu(fc1)
    fc1_drop = tf.nn.dropout(fc1_relu, dropout_keep_rate)

    # fc layer 2
    fc2_weights = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=0.01),name='fc2_weights')
    fc2_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),name='fc2_biases')
    fc2 = tf.matmul(fc1_drop, fc2_weights)
    fc2 = tf.nn.bias_add(fc2, fc2_biases)
    fc2_relu = tf.nn.relu(fc2)
    fc2_drop = tf.nn.dropout(fc2_relu, dropout_keep_rate)

    # fc layer 3
    fc3_weights = tf.Variable(tf.random_normal([4096, M*r], dtype=tf.float32, stddev=0.01),name='fc3_weights')
    fc3_biases = tf.Variable(tf.constant(1.0, shape=[M*r], dtype=tf.float32),name='fc3_biases')
    fc3 = tf.matmul(fc2_drop, fc3_weights)
    u_predict = tf.nn.bias_add(fc3, fc3_biases)
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



if __name__ == '__main__':
    main()
