import os
import glob
import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

%matplotlib inline

# ensures same initialization for neural network
# tf.set_random_seed(1)
tf.reset_default_graph()

oimg_h = 1600
oimg_w = 1200
nimg_w = 256
nimg_h = 256

def toSmall(points):

	for k in range(len(points)):
		for i in range(0,16,2):
			points[k][i] = int(points[k][i]/oimg_h * nimg_h)
			points[k][i+1] = int(points[k][i+1]/oimg_w * nimg_w)

	return points

def toOrig(points):

	for k in range(len(points)):
		for i in range(0,16,2):
			points[k][i] = int(points[k][i]/nimg_h * oimg_h)
			points[k][i+1] = int(points[k][i+1]/nimg_w * oimg_w)

	return points

# Scale range of input to [0,1], orig coords
def normalize(points):
	points = points.astype(float)

	for k in range(len(points)):
		for i in range(0,16,2):
			points[k][i] /= oimg_h / (1 - -1)
			points[k][i] += -1

			points[k][i+1] /= oimg_w / (1 - -1)
			points[k][i+1] += -1

	return points

def rescale(points):

    for k in range(len(points)):
        for i in range(0,16,2):
            points[k][i] += 1
            points[k][i] *= oimg_h

            points[k][i] += 1
            points[k][i] *= oimg_w

    return points

def shuffle_in_unison(a, b, c):
    
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    print('dataset shuffled')
    
    return a, b, c

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# add tf.nn.atrous_conv2d and posenet
def network(img, reuse=False):
    # network variables
    with tf.variable_scope('network', reuse=reuse):
        
        # cnet = tf.nn.atrous_conv2d(cnet, 256, 2, padding='same')
        # cnet = tf.reshape(img, [-1, nimg_h, nimg_w, 1])
        
        # 256
        cnet = tf.layers.conv2d(img, 32, 8, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 32, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)
        
        # 128
        cnet = tf.layers.conv2d(cnet, 64, 3, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 64, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)
        # 64
        cnet = tf.layers.conv2d(cnet, 128, 3, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 128, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)
        # 32
        cnet = tf.layers.conv2d(cnet, 256, 3, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 256, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)
        # 16
        cnet = tf.layers.conv2d(cnet, 512, 3, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 512, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)
        # 8
        cnet = tf.layers.conv2d(cnet, 1024, 3, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 1024, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)
#         cnet = tf.reduce_max(cnet, reduction_indices=[3], keep_dims=True)
        # 4
        cnet = tf.layers.conv2d(cnet, 2048, 3, padding='same', activation='relu')
        cnet = tf.layers.conv2d(cnet, 2048, 3, strides=1, padding='same', activation='relu')
        cnet = tf.layers.conv2d_transpose(cnet, 512, 3, strides=2, padding='same')
        # 16
        cnet = tf.layers.conv2d_transpose(cnet, 256, 3, strides=2, padding='same')
#         cnet = tf.layers.conv2d_transpose(cnet, 64, 3, strides=1, padding='same')
        # 32
        cnet = tf.layers.conv2d_transpose(cnet, 128, 3, strides=2, padding='same')
#         cnet = tf.layers.conv2d_transpose(cnet, 16, 3, strides=1, padding='same')
        # 64
        cnet = tf.layers.conv2d_transpose(cnet, 64, 3, strides=2, padding='same')
        cnet = tf.layers.conv2d_transpose(cnet, 8, 3, strides=2, padding='same')

        return cnet

# save filepath to variable for easier access
path = '../input/wing-dataset/truewing.csv'
# read the data and store data in DataFrame
data = pd.read_csv(path)
points = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
				 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']
name = ['filename']
X = data[points]
files = data[name]
numdata = X.values.tolist()
filenames = files.values.tolist()

numdata = np.asarray(numdata)

### Scale coord test
# img = Image.open('./pics/' + filenames[0][0])
# img2 = np.asarray(img)
# img3 = np.asarray(img)

# for i in range(0,16,2):
# 	cv.circle(img2, (int(numdata[0][i]),int(numdata[0][i+1])), 8, (255,0,0), 3)

# img2 = Image.fromarray(img2)
# img2.show()

# img3 = imutils.resize(img3, width=400)
# for i in range(0,16,2):
# 	x = int(int(numdata[0][i])/1600 * 400)
# 	y = int(int(numdata[0][i+1])/1200 * 300)
# 	cv.circle(img3, (x,y), 4, (255,0,0), 2)

# img3 = Image.fromarray(img3)
# img3.show()
######################

# points = toSmall(numdata)

points = normalize(numdata)
# print(points)
# print(np.max(points))
# print(np.min(points))

testimg = np.load("../input/wing-dataset/imgdataC_256-3.npy")
# testimg = np.load("../input/wing-dataset/imgdataC_256.npy")
testimg = testimg[0:20] / 255
imgdataOrig = np.load("../input/wing-dataset/AugimgdataC_256.npy")
imgdata = np.copy(imgdataOrig) / 255

# hm = np.load("../input/wing-dataset/Augheatmap8_s5_64.npy")
hm = np.load("../input/wing-dataset/Augheatmap8_s3_128.npy")

# why did this fuck me?
# OldRange = (np.max(hm) - np.min(hm))  
# NewRange = (1 - (-1))  
# hm = (((hm - np.min(hm)) * NewRange) / OldRange) + -1

plt.imshow(hm[176,:,:,0])
plt.show()

# XtrainOrig = imgdataOrig[10:3088]
# Xtrain = imgdata[10:3088]
# Ytrain = hm[10:3088]

# XtestOrig = imgdataOrig[0:10]
# Xtest = imgdata[0:10]
# Ytest = hm[0:10]

XtrainOrig = imgdataOrig[10:3088]
Xtrain = imgdata[10:3088]
Ytrain = hm[10:3088]

XtestOrig = imgdataOrig[0:10]
Xtest = imgdata[0:10]
Ytest = hm[0:10]

# for k in range(10):
#     simg = np.asarray(Image.fromarray(imgdataOrig[k]).convert("RGBA"))
#     for j in range(0,16,2):
#         x = int((400/(1-(-1))) * (points[k][j] - 1.0) + 400)
#         y = int((300/(1-(-1))) * (points[k][j+1] - 1.0) + 300)
#         cv.circle(simg, (x,y), 3, (255,0,0), 2)
#     simg = Image.fromarray(simg)
#     simg.show()

# Build the neural network
graph = tf.Graph()
with graph.as_default():
    # our image input size
    image = tf.placeholder(tf.float32, shape=[None, nimg_h, nimg_w, 3])
    # our labels, 1hot
    expected = tf.placeholder(tf.float32, shape=[None, 128, 128, 8])
    
    # Passes image to network and gets output
    prediction = network(image)
    # Compare prediction to truth to calc error
    # network_loss = tf.losses.sigmoid_cross_entropy(expected, prediction)
    network_loss = tf.losses.mean_squared_error(expected, prediction)
#     network_loss = tf.losses.huber_loss(expected, prediction)
    # How much we update network
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001)
    # Create variables
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
    # Update weights of network based off error to improve it
    train = optimizer.minimize(network_loss, var_list=net_vars)
    init = tf.global_variables_initializer()

if not os.path.exists('./network/'):
    os.makedirs('./network/')

with tf.Session(graph=graph) as sess:
    sess.run(init)
    saver = tf.train.Saver()
    
    tf.get_default_graph().finalize()
    
    # amount of images we pass at once
    batch_size = 38
    
    # Each loop goes over entire dataset
    for epoch in range(40):
        
        print("Epoch: %i" % epoch)
        # save our model after every epoch
        if epoch != 0:
            saver.restore(sess, "./network/encoder.ckpt")

        Xtrain, XtrainOrig, Ytrain = shuffle_in_unison(Xtrain, XtrainOrig, Ytrain)

        # Each loop is the batch size of the entire dataset
        for step in range(81):
            # grab new dataset indices
            beg = step*batch_size
            end = (step+1)*batch_size
            
            img_batch = Xtrain[beg:end]
            label_batch = Ytrain[beg:end]
#             label_batch = Ytrain[beg:end,:,:,0:1]
            
#             img_batch = imgdata[176:177]
#             label_batch = hm[176:177]
#             img_batch = hm[176:177,:,:,0:1]
#             label_batch = hm[176:177,:,:,0:1]

            # pass labels and images to network, tells us loss
            feed_dict = {expected: label_batch, image: img_batch}
            _, loss = sess.run([train,network_loss], feed_dict=feed_dict)
            
            print('Step: %i, Loss: %f' % (step,loss))
                
        # save_path = saver.save(sess, "./network/encoder.ckpt")
        # print("Model saved in path: %s" % save_path)
    
#     load_path = saver.restore(sess, "./network/encoder.ckpt")
    # load_path = saver.restore(sess, "../input/wing-model/encoder.ckpt")
    # print("Model loaded from: %s" % load_path)
    
#     Xtrain, XtrainOrig, Ytrain = shuffle_in_unison(Xtrain, XtrainOrig, Ytrain)
    
    for i in range(5):
#         simg = np.copy(Xtest[i:i+1])
#         simg = np.copy(imgdata[176:177])
        simg = np.copy(testimg[i:i+1])
        sample = sess.run(prediction, feed_dict={image: simg})
        # print(sample)
        # sample *= 255
        
#         ehm = np.zeros((64,64), dtype=np.float32)
        ehm = np.zeros((128,128), dtype=np.float32)
        for j in range(7):
            ehm += (sample[0,:,:,j] + 1) / 2

        ehm *= 255
        plt.imshow(simg[0])
        plt.show()
        plt.imshow(ehm)
        plt.show()
        
#     print(hm.shape)
    sample = sess.run(prediction, feed_dict={image: testimg})
    np.save("testhm", sample)
#     simg = np.copy(Xtest[0:10])
#     sample = sess.run(prediction, feed_dict={image: simg})
#     np.save("testhm", sample)