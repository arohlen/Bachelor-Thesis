import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import csv
import cv2
from sklearn.utils import shuffle
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = conv2d(conv1, weights['wc2'], biases['bc2'])

    #conv1 = tf.nn.relu(conv1)

    print("conv1 before pool ",conv1.shape)

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 16*16 matrix.
    conv1 = maxpool2d(conv1, k=2)
    print("conv1 after pool ",conv1.shape)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = conv2d(conv2, weights['wc3'], biases['bc3'])

    #conv2 = tf.nn.relu(conv2)

    print("conv2 before pool ",conv2.shape)

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 8*8 matrix.
    conv2 = maxpool2d(conv2, k=2)

    print("conv2 after pool ",conv2.shape)


    conv3 = conv2d(conv2, weights['wc4'], biases['bc4'])
    conv3 = conv2d(conv3, weights['wc5'], biases['bc5'])

    #conv3 = tf.nn.relu(conv3)

    print("conv3 before pool ",conv3.shape)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)

    print("conv3 after pool ",conv3.shape)


    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    #fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    #fc2 = tf.nn.relu(fc2)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

training_file = '../traffic-signs-data/train.p'
testing_file = '../traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

train_x = train['features']
train_y = train['labels']

train_x, train_y = shuffle(train_x, train_y)

train_y = tf.keras.utils.to_categorical(train_y,43)


test_x = test['features']
test_y = test['labels']

test_y = tf.keras.utils.to_categorical(test_y,43)

#train_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in train_x]), 3)
#test_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in test_x]), 3)

training_iters = 200
learning_rate = 0.001

print("LEARNING RATE",learning_rate)
batch_size = 128

n_input = 32

n_classes = 43

top_acc = 0

x = tf.placeholder("float", [None, 32,32,3])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,3,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()),

    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc4': tf.get_variable('W3', shape=(3,3,128,128), initializer=tf.contrib.layers.xavier_initializer()),

    'wc5': tf.get_variable('W4', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc6': tf.get_variable('W5', shape=(3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()),

    'wd1': tf.get_variable('W6', shape=(4*4*256,128), initializer=tf.contrib.layers.xavier_initializer()),
    #'wd2': tf.get_variable('W7', shape=(128,64), initializer=tf.contrib.layers.xavier_initializer()),

    'out': tf.get_variable('W8', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),

    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),

    'bc5': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bc6': tf.get_variable('B5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),

    'bd1': tf.get_variable('B6', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    #'bd2': tf.get_variable('B7', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),

    'out': tf.get_variable('B8', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_x)//batch_size):
            batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            #opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            opt, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        rand = random.randint(0,10500)
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_x[rand:(rand+2000)],y : test_y[rand:(rand+2000)]})
        test_loss.append(valid_loss)
        test_accuracy.append(test_acc)

        if test_acc > top_acc:
            top_acc = test_acc

        train_loss.append(loss)

        train_accuracy.append(acc)

        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()



plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.savefig('loss.png')
plt.close()

plt.ylim(0,1.1)
plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.legend()
plt.savefig('acc.png')
print("top acc")
print(top_acc)
