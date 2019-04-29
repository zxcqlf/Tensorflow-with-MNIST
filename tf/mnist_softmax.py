import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhaocq/桌面/tensorflow/raw/",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(100):
    batch_xs, label_ys = mnist.train.next_batch(60)
    sess.run(train_step,feed_dict={x:batch_xs,y_:label_ys})
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))


# writer = tf.summary.FileWriter('/home/rainymelody/桌面/PycharmProjects/Tensorflow/model',
#                                tf.get_default_graph())
# writer.close()
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
