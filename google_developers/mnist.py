# we are going to use tensorflow for poets
# which is really good at image classification
# the only thing we need is training data
# we are using tensorflow's flower data set for this
# tensorflow deals with deep learning
# deep learning extracts features automatically, not manually
# for example, in iris, each column is a feature, and we came
# up with these features manually
# in tensorflow, the data is just the list of labeled images
# here x is a 2D array of pixels
# y is the label, like rows
# in deep learning, the classifier we use is called a neural network
# neural networks can learn more complex functions
# tf learn is a high level ml library on top of tensorflow
# it is similar to sklearn

# let's download the mnist data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# import the data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# be able to input any number of mnist images
# there are 784 pixels in one mnist image
x = tf.placeholder(tf.float32, [None,784])

# in tensorflow, a variable is a modifiable tensor
# that lives in tensorflow's interacting operations
# we will introduce weights W and biases b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# it takes one line fo implement our model (define it)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# now we have to train our data
# in machine learning, we can use a model called cross-entropy
# to figure out what our loss is

# add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# implement cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




