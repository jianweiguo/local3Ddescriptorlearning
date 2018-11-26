
# coding: utf-8

# In[1]:


import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


# In[2]:


parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpuid', '-g', default='0', type=str, metavar='N',
                    help='GPU id to run') 

parser.add_argument('--learning_rate', '--lr', default=0.0003, type=float, 
                    help='the learning rate')
parser.add_argument('--l2_regularizer_scale', default=0.005, type=float,
                    help='scale parameter used in the l2 regularization')
parser.add_argument('--n_iteration', '-n', default=100000, type=int,metavar='N',
                    help='number of training iterations')

parser.add_argument('--batch_size', '--bs', default=128, type=int, 
                    help='size of training batch, it is equal to batch_keypoint_num*batch_gi_num')
parser.add_argument('--batch_keypoint_num', '--bkn', default=16, type=int, 
                    help='number of different keypoints in a training batch')
parser.add_argument('--batch_gi_num', '--bgn', default=8, type=int,
                    help='number of geometry images of one keypoint in a training batch')

parser.add_argument('--val_freq', '--vf', default=5, type=int,
                    help='frequency of validation.')
parser.add_argument('--print_freq', '--pf', default=100, type=int,
                    help=r'print info every {print_freq} iterations')
parser.add_argument('--save_freq', '--sf', default=500, type=int,
                    help=r'save the current trained model every {save_freq} iterations')

parser.add_argument('--summary_saving_dir', '--ssd', default='./summary', type=str, 
                    help='directory to save summaries')
parser.add_argument('--model_saving_dir', '--msd', default='./saved_models', type=str,
                    help='directory to save trained models')

parser.add_argument('--restore', '-r', dest='restore',default=True, action='store_true',
                    help='bool value, restore variables from saved model of not')
parser.add_argument('--restore_path',default='/data/yqwang/Project/3dDescriptor/train_softmax_adam/saved_models/training_model_multiuse-9999', 
                    type=str, 
                    help='path to the saved model(if restore)')

parser.add_argument('--use_kpi_set', dest='use_kpi_set', default=True, action='store_true', 
                    help='bool value, use keypoint set from keypoint file or not')
parser.add_argument('--keypoints_path', default='/data/yqwang/Dataset/faust_256p/keypoints_faust256.kpi', 
                    type=str, 
                    help='path to the keypoint file(if use_kpi_set)')
parser.add_argument('--n_all_points', default=6890, type=int, 
                    help='number of all points in the model')


parser.add_argument('--shuffle_batch_capacity', default=400, type=int,
                    help='capacity of shuffle bacth buffer')
parser.add_argument('--gi_size', default=32, type=int,
                    help='length and width of geometry image, assuming it\'s square')
parser.add_argument('--gi_channel', default=31, type=int,
                    help='number of geometry image channels')
parser.add_argument('--triplet_loss_gap', default=1, type=float,
                    help='the gap value used in the triplet loss')
parser.add_argument('--n_loss_compute_iter', default=17, type=int,
                    help='number of iterations to compute the training loss')
# parser.add_argument('--n_test_iter', default=100, type=int,
#                     help='number of iterations to compute the test results')
parser.add_argument('--tfr_dir', default='/data/yqwang/Dataset/faust_256p/gi_TFRecords', 
                    type=str, 
                    help='directory of training TFRecords, containing' 
                         '3 subdirectories: \"train\", \"val\", and \"test\"')
parser.add_argument('--tfr_name_template', default=r'pidx_%04d.tfrecords', type=str, 
                    help='name template of TFRecords filenames')

global args


# In[3]:


def read_index_file(path, delimiter=' '):
    """
    Read indices from a text file and return a list of indices.
    :param path: path of the text file.
    :return: a list of indices.
    """

    index_list = []
    with open(path, 'r') as text:

        for line in text:
            ls = line.strip(' {}[]\t')

            if not ls or ls[0] == '#':  # Comment content
                continue
            ll = ls.split(delimiter)

            for id_str in ll:
                idst = id_str.strip()
                if idst == '':
                    continue
                index_list.append(int(idst))

    return index_list


# In[4]:


def append_log(path, string_stream):
    """
    Write string_stream in a log file.
    :param path: path of the log file.
    :param string_stream: string that will be write.
    """

    with open(path, 'a') as log:
        log.write(string_stream)
    return


# In[5]:


class TripletNet:
    def __init__(self, args=None, is_training=True):
        self.args = args
        self.is_training = is_training
        # self.predict_net =None
        self.anchor_net = None  # anchor_net is also the predict_net
        self.positive_net = None
        self.negative_net = None
        self.descriptors = None  # descriptors of anchors
        self.cost = None
        self.cost_same = None
        self.cost_diff = None
        self.all_multiuse_params = None
        self.predictions = None
        self.acc = None

    def inference(self, gi_placeholder, reuse=None):  # reuse=None is equal to reuse=False(i.e. don't reuse)
        with tf.variable_scope('model', reuse=reuse):
            tl.layers.set_name_reuse(reuse)  # reuse!

            network = tl.layers.InputLayer(gi_placeholder, name='input')

            """ conv2 """
            network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv2_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn2_1')

            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                padding='SAME', name='pool2')

            """ conv3 """
            network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv3_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn3_1')

            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                padding='SAME', name='pool3')

            """ conv4 """
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv4_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn4_1')

            network = MeanPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                 padding='SAME', name='pool4')


            network = FlattenLayer(network, name='flatten')
            network = DenseLayer(network, n_units=512, act=tf.identity, name='fc1_relu')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn_fc')
            # network = DenseLayer(network, n_units=4096, act=args.activation, name='fc2_relu')
            # network = DenseLayer(network, n_units=10, act=tf.identity, name='fc3_relu')
            network = DenseLayer(network, n_units=128, act=tf.identity, name='128d_embedding')

        return network


    def build_nets(self, anchor_placeholder, positive_placeholder, negative_placeholder, anchor_label_placeholder, keypoint_num):
        self.anchor_net = self.inference(anchor_placeholder, reuse=None)
        self.positive_net = self.inference(positive_placeholder, reuse=True)
        self.negative_net = self.inference(negative_placeholder, reuse=True)

#         gap = tf.constant(np.float32(args.triplet_loss_gap))
# #         zero = tf.constant(np.float32(0))
#         zeros = tf.zeros(args.batch_size)


        self.all_multiuse_params = self.anchor_net.all_params.copy()

#         self.anchor_net.outputs = args.activation(self.anchor_net.outputs)

#         self.anchor_net = DenseLayer(self.anchor_net, n_units=keypoint_num, act=tf.identity, name='feature')

#         logits = self.anchor_net.outputs
#         self.predictions = tf.nn.softmax(logits)
#         self.cost = tl.cost.cross_entropy(output=logits, target=anchor_label_placeholder, name='cost')

#         correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), anchor_label_placeholder)
#         self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')


        self.cost_same = tf.norm(self.anchor_net.outputs - self.positive_net.outputs, axis=1)
        print(self.cost_same.shape)
#         assert self.cost_same.shape[0] == args.batch_size
        self.cost_diff = tf.norm(self.anchor_net.outputs - self.negative_net.outputs, axis=1)

        delta_mean = tf.reduce_mean(self.cost_same - self.cost_diff)
        cost_same_mean, cost_same_variance = tf.nn.moments(self.cost_same, axes=[0])
        cost_diff_mean, _ = tf.nn.moments(self.cost_diff, axes=[0])
        
        cv = tf.sqrt(cost_same_variance)/cost_same_mean

        # self.cost = tf.maximum(zero, gap + delta)
        # ratio = self.cost_diff / self.cost_same
        # self.cost = - (gap + self.cost_diff) / (gap + self.cost_same) + self.cost_same

        ratio = cost_diff_mean / cost_same_mean

#         batch_variance = args.batch_size*tf.sqrt(cost_same_variance)

        # self.cost = self.cost_same + tf.maximum(zero, gap - self.cost_diff) +  batch_variance
        self.cost = tf.reduce_sum(tf.maximum(np.float32(0), np.float32(args.triplet_loss_gap) + self.cost_same - self.cost_diff)) + 0.1*args.batch_size*cv  

        tf.summary.scalar(name='cost', tensor=self.cost)
        tf.summary.scalar(name='delta_mean', tensor=delta_mean)
        tf.summary.scalar(name='ratio', tensor=ratio)
        tf.summary.scalar(name='cost_same_mean', tensor=cost_same_mean)
        tf.summary.scalar(name='cost_diff_mean', tensor=cost_diff_mean)
#         tf.summary.scalar(name='batch_variance', tensor=batch_variance)
        tf.summary.scalar(name='CV', tensor=cv)
        # tf.summary.scalar(name='accuracy', tensor=self.acc)

        # Weight decay
        l2 = 0
        for p in tl.layers.get_variables_with_name('W_conv2d'):
            l2 += tf.contrib.layers.l2_regularizer(args.l2_regularizer_scale)(p)
            tf.summary.histogram(name=p.name, values=p)

        for p in tl.layers.get_variables_with_name('128d_embedding/W'):
            l2 += tf.contrib.layers.l2_regularizer(args.l2_regularizer_scale)(p)
            tf.summary.histogram(name=p.name, values=p)

        self.cost += l2


        # print(len(tl.layers.get_variables_with_name('128d_embedding/W')))
        #
        # print('--------------------------------------------------------------------------------')
        # print(len(listv))

        # self.cost = tf.maximum(zero, gap - tf.norm(self.anchor_net.outputs - self.negative_net.outputs))
        # self.cost = tf.maximum(zero, gap + tf.norm(self.anchor_net.outputs - self.positive_net.outputs))
        # self.cost = tf.maximum(tf.constant(np.float32(-100000)), gap - tf.norm(self.anchor_net.outputs - self.negative_net.outputs))

        # self.cost = self.cost + tl.cost.maxnorm_regularizer(1.0)(self.network.all_params)
        # self.cost = self.cost + tf.contrib.layers.l2_regularizer(1.0)(self.anchor_net.all_params)


# In[6]:


args = parser.parse_args(args=['-g 2'])
setattr(args, 'conv_initializer', tf.contrib.layers.xavier_initializer())
setattr(args, 'activation', tl.activation.leaky_relu)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
train_tfr_dir = join(args.tfr_dir, 'train')
val_tfr_dir = join(args.tfr_dir, 'val')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

if args.use_kpi_set:
    keypoint_list = read_index_file(args.keypoints_path) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
else:
    keypoint_list = list(range(args.n_all_points))

# debug

# keypoint_list = list(range(16))

keypoint_num = len(keypoint_list)

# rebuild 0-based index
keypoint_list = list(range(keypoint_num))


# In[7]:


def parse_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'gi_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })

    gi = tf.decode_raw(features['gi_raw'], tf.float32)
    gi = tf.reshape(gi, [args.gi_size, args.gi_size, args.gi_channel])
    label = tf.cast(features['label'], tf.int32)  # throw label tensor
    return gi, label


# In[8]:


run_time = time.localtime(time.time())

# [batch_size, height, width, channels]
anchor_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, args.gi_size, args.gi_size, args.gi_channel])  # [batch_size, height, width, channels]

positive_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, args.gi_size, args.gi_size, args.gi_channel])  # [batch_size, height, width, channels]

negative_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, args.gi_size, args.gi_size, args.gi_channel])  # [batch_size, height, width, channels]

anchor_label_placeholder = tf.placeholder(
    dtype=tf.int32,
    shape=[None])  # [batch_size, height, width, channels]

triplet_net = TripletNet(is_training=True) # training 

triplet_net.build_nets(
    anchor_placeholder=anchor_placeholder,
    positive_placeholder=positive_placeholder,
    negative_placeholder=negative_placeholder,
    anchor_label_placeholder=anchor_label_placeholder,
    keypoint_num=keypoint_num
)

train_params = triplet_net.anchor_net.all_params

# train_op = tf.train.AdamOptimizer(args.learning_rate, beta1=0.9, beta2=0.999,
#                                   epsilon=1e-08, use_locking=False).minimize(triplet_net.cost,
#                                                                              var_list=train_params)
# # train_op = tf.train.AdadeltaOptimizer().minimize(triplet_net.cost, var_list=train_params)

# Summary for visualization.
# tf.summary.scalar(name='cost', tensor=triplet_net.cost)
# tf.summary.scalar(name='cost_same', tensor=triplet_net.cost_same)
# tf.summary.scalar(name='cost_diff', tensor=triplet_net.cost_diff)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(join(args.summary_saving_dir, 'train'), sess.graph)
validation_writer = tf.summary.FileWriter(join(args.summary_saving_dir, 'validation'), sess.graph)
log_path = join('./', 'log_' + time.strftime('%Y-%m-%d_%H-%M-%S', run_time) + '.log')
log_stream = 'Start running at ' + time.strftime('%Y-%m-%d %H:%M:%S', run_time) + '.\n'
log_stream += '================================================================================\n'

with open(log_path, 'w') as logf:
    logf.write(log_stream)

log_stream = ''

temp = set(tf.global_variables())

if args.restore:
    tl.layers.initialize_global_variables(sess)
    load_saver = tf.train.Saver()
    load_saver.restore(sess, args.restore_path)
    info = 'Restore model parameters from %s' % args.restore_path
    log_stream += info
    log_stream += '\n'
    print(info)

else:
    tl.layers.initialize_global_variables(sess)
    info = 'Successfully initialized global variables.'
    log_stream += info
    log_stream += '\n'
    print(info)       

train_op = tf.train.AdamOptimizer(args.learning_rate, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(triplet_net.cost,
                                                                             var_list=train_params)
sess.run(tf.initialize_variables(set(tf.global_variables()) - temp))
# train_op = tf.train.AdadeltaOptimizer().minimize(triplet_net.cost, var_list=train_params)
    
triplet_net.anchor_net.print_params()
triplet_net.anchor_net.print_layers()

info = '   learning_rate: %f' % args.learning_rate
log_stream += info
log_stream += '\n'
print(info)

info = '   batch_size: %d' % args.batch_size
log_stream += info
log_stream += '\n'
print(info)

append_log(log_path, log_stream)
log_stream = ''


# In[9]:


# data generator preperation
# start_time = time.time()

# train_iter_batch_list = []
# val_iter_batch_list = []
# test_iter_batch_list = []

# for i in range(keypoint_num):
#     train_tfrecords_path = join(train_tfr_dir, args.tfr_name_template % i)
#     val_tfrecords_path = join(val_tfr_dir, args.tfr_name_template % i)
# #         test_tfrecords_path = join(args.test_tfr_dir, args.tfr_name_template % i)

#     train_iter_batch_list.append(
#         tf.data.TFRecordDataset(train_tfrecords_path).map(parse_and_decode). \
#         shuffle(args.shuffle_batch_capacity).batch(args.batch_gi_num).repeat().make_one_shot_iterator().get_next())

#     val_iter_batch_list.append(
#         tf.data.TFRecordDataset(val_tfrecords_path).map(parse_and_decode). \
#         shuffle(args.shuffle_batch_capacity).batch(args.batch_gi_num).repeat().make_one_shot_iterator().get_next())
    
#     if (i + 1) % 100  == 0:
#         print('%4d of %d are processed. ' % ((i + 1), keypoint_num))

# #         test_iter_batch_list.append(
# #             tf.data.TFRecordDataset(test_tfrecords_path).map(parse_and_decode). \
# #             shuffle(args.shuffle_batch_capacity).batch(args.batch_gi_num).repeat().make_one_shot_iterator().get_next())

# info = 'Data generator preperation costs %fs' % (time.time() - start_time)
# log_stream += info
# log_stream += '\n'
# print(info)

# append_log(log_path, log_stream)
# log_stream = ''

# tfr_path_placeholder = tf.placeholder(tf.string, shape=[])


# batch_iterator = tf.data.TFRecordDataset(tfr_path_placeholder).map(parse_and_decode). \
#                  shuffle(args.shuffle_batch_capacity).batch(args.batch_gi_num).repeat().make_initializable_iterator()
# next_batch = batch_iterator.get_next()

# setattr(args, 'n_tfr_iteration', 100)
# setattr(args, 'tfr_batch_size', 128)


train_placeholder_list = []
val_placeholder_list = []

train_iter_list = []
val_iter_list = []

train_next_element_list = []
val_next_element_list = []

for keypoint_idx in range(keypoint_num):
#     train_tfr_path = tf.placeholder(tf.string, shape=[])
#     train_placeholder_list.append(train_tfr_path)
    
    train_iter = tf.data.TFRecordDataset(join(train_tfr_dir, args.tfr_name_template % keypoint_idx)).map(parse_and_decode).                  shuffle(args.shuffle_batch_capacity).batch(args.batch_gi_num).repeat().make_one_shot_iterator()
    train_iter_list.append(train_iter)
    
    train_next_element = train_iter.get_next()
    train_next_element_list.append(train_next_element)
    
#     val_tfr_path = tf.placeholder(tf.string, shape=[])
#     val_placeholder_list.append(val_tfr_path)
    
    val_iter = tf.data.TFRecordDataset(join(val_tfr_dir, args.tfr_name_template % keypoint_idx)).map(parse_and_decode).                  shuffle(args.shuffle_batch_capacity).batch(args.batch_gi_num).repeat().make_one_shot_iterator()
    val_iter_list.append(val_iter)
    
    val_next_element = val_iter.get_next()
    val_next_element_list.append(val_next_element)


# In[ ]:


# training

start_time = time.time()

# pidx2position = dict()
# loaded_keypoint_list = None

for iteration in range(args.n_iteration):
    
#     if iteration % args.n_tfr_iteration == 0:
#         ls = time.time()
        
#         loaded_keypoint_list = np.random.choice(a=keypoint_list, size=args.tfr_batch_size, replace=False)
#         pidx2position.clear()
        
#         for n, keypoint_idx in enumerate(loaded_keypoint_list):
#             sess.run(train_iter_list[n].initializer, feed_dict={train_placeholder_list[n]: join(train_tfr_dir, args.tfr_name_template % keypoint_idx)})
#             sess.run(val_iter_list[n].initializer, feed_dict={val_placeholder_list[n]: join(val_tfr_dir, args.tfr_name_template % keypoint_idx)})
#             pidx2position[keypoint_idx] = n 
        
#         print('loading tfr batch  time cost: %f' % (time.time() - ls))
        
    selected_keypoints = np.random.choice(a=keypoint_list, size=args.batch_keypoint_num, replace=False)

    
    ss = time.time()
    train_anchor_gi_all = None
    train_label_all = None

    for keypoint_id in selected_keypoints:
        
       
        train_gi, train_label = sess.run(train_next_element_list[keypoint_id])

        if train_anchor_gi_all is None:
            train_anchor_gi_all = train_gi
        else:
            train_anchor_gi_all = np.append(train_anchor_gi_all, train_gi, axis=0)

        if train_label_all is None:
            train_label_all = train_label
        else:
            train_label_all = np.append(train_label_all, train_label, axis=0)
            
    print('select train tfr time cost: %f' % (time.time() - ss))
    # triplet_net.is_training = False  # compute descriptors

    triplet_net.is_training = True  # train
    
    train_positive_gi_all = np.zeros_like(train_anchor_gi_all)
    train_negative_gi_all = np.zeros_like(train_anchor_gi_all)
    
    for index in range(len(train_anchor_gi_all)):

        gi_same = np.asarray([target_gi for j, target_gi in enumerate(train_anchor_gi_all)
                              if train_label_all[j] == train_label_all[index] and j != index])
        train_positive_gi_all[index] = gi_same[np.random.choice(a=range(len(gi_same)), size=1)[0]]

        gi_diff = np.asarray([target_gi for j, target_gi in enumerate(train_anchor_gi_all)
                              if train_label_all[j] != train_label_all[index]])
        train_negative_gi_all[index] = gi_diff[np.random.choice(a=range(len(gi_diff)), size=1)[0]]
    
    
    ts = time.time()
#     _, summary = sess.run([train_op, merged_summary],
#                           feed_dict={anchor_placeholder: train_anchor_gi_all,
#                                      anchor_label_placeholder: train_label_all})


    _, summary = sess.run([train_op, merged_summary],
                          feed_dict={anchor_placeholder: train_anchor_gi_all,
                                     positive_placeholder: train_positive_gi_all,
                                     negative_placeholder: train_negative_gi_all})


    print('train time cost: %f' % (time.time() - ts))

    train_writer.add_summary(summary, global_step=iteration)

    if iteration % args.val_freq == 0:

        ss = time.time()
        val_anchor_gi_all = None
        val_label_all = None

        for keypoint_id in selected_keypoints:

            val_gi, val_label = sess.run(val_next_element_list[keypoint_id])

            if val_anchor_gi_all is None:
                val_anchor_gi_all = val_gi
            else:
                val_anchor_gi_all = np.append(val_anchor_gi_all, val_gi, axis=0)

            if val_label_all is None:
                val_label_all = val_label
            else:
                val_label_all = np.append(val_label_all, val_label, axis=0)

        print('select val tfr time cost: %f' % (time.time() - ss))

        val_positive_gi_all = np.zeros_like(val_anchor_gi_all)
        val_negative_gi_all = np.zeros_like(val_anchor_gi_all)

        for index in range(len(val_anchor_gi_all)):
            gi_same = np.asarray([target_gi for j, target_gi in enumerate(val_anchor_gi_all)
                                  if val_label_all[j] == val_label_all[index] and j != index])
            val_positive_gi_all[index] = gi_same[np.random.choice(a=range(len(gi_same)), size=1)[0]]

            gi_diff = np.asarray([target_gi for j, target_gi in enumerate(val_anchor_gi_all)
                                  if val_label_all[j] != val_label_all[index]])
            val_negative_gi_all[index] = gi_diff[np.random.choice(a=range(len(gi_diff)), size=1)[0]]  


        vs = time.time()
    #     summary = sess.run(merged_summary, feed_dict={anchor_placeholder: val_anchor_gi_all,
    #                                                   anchor_label_placeholder: val_label_all})
        summary = sess.run(merged_summary, feed_dict={anchor_placeholder: val_anchor_gi_all,
                                                      positive_placeholder: val_positive_gi_all,
                                                      negative_placeholder: val_negative_gi_all})
        print('val time cost: %f' % (time.time() - vs))

        validation_writer.add_summary(summary, global_step=iteration)
        
        print('!!!!!!!!----------current iteration: %d'%iteration)


#     if iteration == 0 or (iteration + 1) % args.print_freq == 0:
#         # Calculate train loss and validation loss.
#         info = 'Iteration %d of %d took %fs from last displayed iteration.' % \
#                (iteration + 1, args.n_iteration, time.time() - start_time)
#         log_stream += info
#         log_stream += '\n'
#         print(info)
#         start_time = time.time()


#         train_loss, train_acc = \
#             sess.run([triplet_net.cost, triplet_net.acc],
#                      feed_dict={anchor_placeholder: train_anchor_gi_all,
#                                 anchor_label_placeholder: train_label_all})


#         # Compute validation loss.
#         # hard-mining is disabled in this part.
#         val_loss, val_acc = \
#             sess.run([triplet_net.cost, triplet_net.acc],
#                      feed_dict={anchor_placeholder: val_anchor_gi_all,
#                                 anchor_label_placeholder: val_label_all})


#         info = '    train loss: %f' % (train_loss)
#         log_stream += info
#         log_stream += '\n'
#         print(info)

#         info = '    train acc: %f' % (train_acc)
#         log_stream += info
#         log_stream += '\n'
#         print(info)

#         info = '    validation loss: %f' % (val_loss)
#         log_stream += info
#         log_stream += '\n'
#         print(info)

#         info = '    validation acc: %f' % (val_acc)
#         log_stream += info
#         log_stream += '\n'
#         print(info)

#         append_log(log_path, log_stream)
#         log_stream = ''

    if (iteration + 1) % (args.save_freq) == 0:
        saver_part = tf.train.Saver(triplet_net.all_multiuse_params)
        saver_part.save(sess, join(args.model_saving_dir, 'training_model_multiuse'), global_step=iteration)

        saver = tf.train.Saver()
        saver.save(sess, join(args.model_saving_dir, 'training_model'), global_step=iteration)
#         end

