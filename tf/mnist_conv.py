import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/zhaocq/桌面/tensorflow/raw/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

def weights(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w_conv1 = weights([5,5,1,32])
b_conv1 = bias([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = maxpool(h_conv1)

w_conv2 = weights([5,5,32,64])
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = maxpool(h_conv2)

w_fc1 = weights([7*7*64,1024])
b_fc1 = bias([1024])
h_poo2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_poo2_flat,w_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = weights([1024,10])
b_fc2 = bias([10])
y_conv = tf.matmul(h_fc1_drop,w_fc2)+b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_ : batch[1],keep_prob:0.5})
            print("step %d, training accuracy: %g"% (i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    print("test accuracy: %g"% accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))