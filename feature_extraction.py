import time
import tensorflow as tf
import numpy as np
import pandas as pd
import imageio
from alexnet import AlexNet

if tf.__version__ > '2.0':
    print("Installed Tensorflow is not 1.x,it is %s" % tf.__version__)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior() 
    
sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)
# TODO: Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs. Assign the result of the softmax activation to `probs` below.
# HINT: Look at the final layer definition in alexnet.py to get an idea of what this
# should look like.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
probs = tf.nn.softmax(tf.matmul(fc7,tf.Variable(tf.truncated_normal(shape,stddev=1e-2))) + tf.Variable(tf.zeros(nb_classes)))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imageio.imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imageio.imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
