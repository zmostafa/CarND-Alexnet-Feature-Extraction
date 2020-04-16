import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

if tf.__version__ > '2.0':
    print("Installed Tensorflow is not 1.x,it is %s" % tf.__version__)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    train = pickle.load(f)

# print(type(train))
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(train['features'],train['labels'],test_size=0.2)

# TODO: Define placeholders and resize operation.
X = tf.placeholder(tf.float32,(None,32,32,3))
y = tf.placeholder(tf.int64)
resized = tf.image.resize(X,(227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
nb_classes = len(np.unique(train['labels']))
shape = (fc7.get_shape().as_list()[-1], nb_classes)
logits = tf.nn.softmax(tf.matmul(fc7,tf.Variable(tf.truncated_normal(shape,stddev=1e-2))) + tf.Variable(tf.zeros(nb_classes)))

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, 43)

rate = 0.001
EPOCHS = 50
BATCH_SIZE = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={X: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training on %d samples..."  %num_examples)
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={X: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        # print("Validation Loss = {:.3f}".format(loss_operation))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './trafficSigns-Classifier')
    print("Model saved")