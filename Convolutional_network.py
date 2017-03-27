
# coding: utf-8

# In[1]:

import numpy as np
import os
import time
import xml.etree.ElementTree
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from PIL import Image


# In[2]:

data_path = 'dataset/MMI database/Sessions/'
all_sess_dir = os.listdir(data_path)[1:]

labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
proto_emo = ['1','2','3','4','5','6']

# In[3]:

emotions = set()
subjects = set()
sub_dict = {}
vid_lst = []
for sess in all_sess_dir:
    sess_dir = data_path + '/' + sess
    
    xml_path = sess_dir + '/session.xml'
    
    xml_tree = xml.etree.ElementTree.parse(xml_path)
    root = xml_tree.getroot()
    
    sub_node = root.getchildren()[0]
    sub_id = sub_node.get('id')
    
    track_node = root.getchildren()[1]
    annotation_node = track_node.getchildren()[0]
    metafile = annotation_node.get('filename')
    
    meta_tree = xml.etree.ElementTree.parse(sess_dir + '/' + metafile)
    meta_root = meta_tree.getroot()
    emo_node = meta_root.getchildren()[1]
    emo_val = emo_node.get('Value')
    emotions.add(emo_val)
    if emo_val in proto_emo:
        subjects.add(sub_id)
        vid_lst.append(sess)
        if sub_id in sub_dict:
            sub_dict[sub_id].append(sess)
        else:
            sub_dict[sub_id] = list([sess])


# In[4]:

sub_lst = list(subjects)
# shuffle(sub_lst)

# In[5]:

val_ratio = 0.15
test_ratio = 0.1

val_size = int(len(sub_lst)*val_ratio)
test_size = int(len(sub_lst)*test_ratio)
train_size = len(sub_lst) - val_size - test_size

# In[7]:

'''
train_subs = sub_lst[:train_size]
val_subs = sub_lst[train_size:train_size+val_size]
test_subs = sub_lst[train_size+val_size:]
print train_subs
print val_subs
'''


# In[6]:

train_subs = ['39', '46', '42', '5', '41', '2',
              '32', '50', '35', '53', '1', '43',
              '33', '6', '48', '34', '3', '31',
              '40', '54', '44', '30', '16', '49']
val_subs = ['15', '36', '47', '45']
test_subs = ['37', '28', '21']


# In[7]:

count = [0,0,0]
train_sess_lst = []
val_sess_lst = []
test_sess_lst = []
for sub in test_subs:
    count[2] += len(sub_dict[sub])
    test_sess_lst += sub_dict[sub]
    
for sub in val_subs:
    count[1] += len(sub_dict[sub])
    val_sess_lst += sub_dict[sub]
    
for sub in train_subs:
    count[0] += len(sub_dict[sub])
    train_sess_lst += sub_dict[sub]

# In[10]:

path_lst = [data_path + sess for sess in train_sess_lst]


# In[11]:

shuffle(path_lst)


# In[12]:

# Functions for convolutional networks
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # strides=[batch, x, y, channel]

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[13]:

x = tf.placeholder(tf.float32, shape=[None, 144, 180, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 77760])


# # Constructing the convolutional network

# In[14]:

# First layer
W_conv1 = weight_variable([5, 5, 3, 128], name='conv1_w')
b_conv1 = bias_variable([128], name='conv1_b')

#x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
W_conv2 = weight_variable([5, 5, 128, 256], name='conv2_w')
b_conv2 = bias_variable([256], name='conv2_b')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([36 * 45 * 256, 1024], name='fc1_w')
b_fc1 = bias_variable([1024], name='fc1_b')

h_pool2_flat = tf.reshape(h_pool2, [-1, 36*45*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 77760], name='fc2_w')
b_fc2 = bias_variable([77760], name='fc2_b')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[15]:

# Optimization
mse = tf.reduce_mean(tf.square(tf.subtract(y_conv, y_)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# In[29]:

all_files = []
for sess_dir in path_lst:
    all_frames = os.listdir(sess_dir + '/images/')
    all_frames = [sess_dir + '/images/' + frame for frame in all_frames]
    all_files += all_frames

shuffle(all_files)


# In[17]:

before_init = time.time()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()

after_init = time.time()

print 'Initialization time: %f' % (after_init-before_init)

input_img_size = 180,144 #360,288
output_img_size = 180,144 #270,216
batch_size = 50

for i in range(5):
    for j in xrange(len(all_files)/batch_size + 1):
        before_train = time.time()
        
        tmp_frame_lst = []
        if j == len(all_files)/batch_size:
            tmp_frame_lst = all_files[j*batch_size:]
        else:
            tmp_frame_lst = all_files[j*batch_size:(j*batch_size)+batch_size]
            
        frame_seq = []
        output = []
        for a_file in tmp_frame_lst:
            img = Image.open(a_file)
            input_img = img.resize(input_img_size, Image.ANTIALIAS)
            pix = np.array(input_img)
            frame_seq.append(pix)
            
            output_img = img.resize(output_img_size, Image.ANTIALIAS)
            pix = np.array(output_img)
            pix = np.reshape(pix, [77760])
            output.append(pix)
            
        frame_seq = np.array(frame_seq)
        output = np.array(output)
        
        if j%45 == 0 and j != 0:
            error_dis = mse.eval(feed_dict={
                x: frame_seq, y_: output, keep_prob: 1.0})
            print("step %d, Mean square error %g"%(i, error_dis))

        train_step.run(feed_dict={x: frame_seq, y_: output, keep_prob: 0.5})
        
        after_train = time.time()
        if j%20 == 0:
            print 'Training time: %f' % (after_train-before_train)

save_path = saver.save(sess, "saved_model/convolutional.ckpt")
print("Model saved in file: %s" % save_path)

