
import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data
import os
 
''''

Link to lecture slides: 
https://docs.google.com/presentation/d/1Vh8NPCWkgVy_I79aqjHyDnlp7TNmVfaEqfIc4LAM-JM/edit?usp=sharing


Tasks:
1. Check that the given implementation reaches 95% test accuracy for
   architecture input-64-64-10 in a few thousand batches.

2. Improve initialization and check that the network learns much faster
   and reaches over 97% test accuracy.

3. Check, that with proper initialization we can train architecture input-64-64-64-64-64-10,
   while with bad initialization it does not even get off the ground.

4. If you do not feel comfortable enough with training networks and/or tensorflow I suggest adding
dropout implemented in tensorflow (check documentation, new placeholder will be needed to indicate train/test phase).

5. Check that with 10 hidden layers (64 units each) even with proper initialization
   the network has a hard time to start learning.

6. Implement batch normalization (use train mode also for testing - it should perform well enough):
    * compute batch mean and variance as tensorflow operations,
    * add new variables beta and gamma to scale and shift the result,
    * check that the networks learns much faster for 5 layers (even though training time per batch is a bit longer),
    * check that the network learns even for 10 hidden layers.

Bonus task:

Design and implement in tensorflow (by using tensorflow functions) a simple convnet and achieve 99% test accuracy.

Note:
This is an exemplary exercise. MNIST dataset is very simple and we are using it here to get resuts quickly.
To get more meaningful experience with training convnets use the CIFAR dataset.
'''


class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.keep_prob: 1.0})
        return results[1:]

    def weight_variable(self, shape, stddev):
        initializer = tf.random_normal(shape, stddev=stddev)
        return tf.Variable(initializer, name='weight')

    def bias_variable(self, shape, bias):
        initializer = tf.constant(bias, shape=shape)
        return tf.Variable(initializer, name='bias')

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 10])

        signal = self.x
        print 'shape', signal.get_shape()



        signal = tf.reshape(signal, [-1, 28, 28, 1])

        conv_depth_list = [64, 32, 16, 8]
        for idx, new_depth in enumerate(conv_depth_list):
            cur_depth = int(signal.get_shape()[3])
            W = self.weight_variable([5, 5, cur_depth, new_depth], 0.1)
            signal = tf.nn.conv2d(signal, W, strides=[1, 1, 1, 1], padding='VALID')

            b = self.bias_variable([new_depth], 0.0)
            signal += b

            # neur_avg = tf.reduce_mean(signal, axis=[0,1,2])
            # neur_stddev = tf.reduce_mean(
            #     (signal - neur_avg) ** 2, axis=[0,1,2]
            # )
            # signal = (signal - neur_avg) / tf.sqrt(neur_stddev + 1e-8);
            #
            # beta = self.bias_variable([new_depth], 0.0)
            # gamma = self.bias_variable([new_depth], 1.0)
            # signal = signal * gamma + beta

            print 'shape', signal.get_shape()

        sh = signal.get_shape()
        signal = tf.reshape(signal, [-1, int(sh[1]*sh[2]*sh[3])])
        neurons_list = [64] * 1 + [10]
        self.keep_prob = tf.placeholder(tf.float32)
        for idx, new_num_neurons in enumerate(neurons_list):
            cur_num_neurons = int(signal.get_shape()[1])
            with tf.variable_scope('fc_'+str(idx+1)):
                W_fc = self.weight_variable([cur_num_neurons, new_num_neurons],
                                            stddev=(math.sqrt(2. / cur_num_neurons)))
                                            # stddev=1.0)

            signal = tf.matmul(signal, W_fc)

            if idx != len(neurons_list)-1:
                neur_avg = tf.reduce_mean(signal, axis=0);
                neur_stddev = tf.reduce_mean(
                    (signal - neur_avg) ** 2,
                    axis=0
                )
                signal = (signal - neur_avg) / tf.sqrt(neur_stddev + 1e-8)
                beta = self.bias_variable([new_num_neurons], 0.0)
                gamma = self.bias_variable([new_num_neurons], 1.0)
                signal = gamma * signal + beta

                signal = tf.maximum(signal / 5., signal)



                # signal = tf.nn.relu(signal)
            # if idx != 0 and idx != len(neurons_list)-1:
                # signal = tf.nn.dropout(signal, self.keep_prob)

            print 'shape', signal.get_shape()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, targets=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        print 'list of variables', map(lambda x: x.name, tf.global_variables())

    def train(self):
 
        self.create_model()
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

 
        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)
 
                    vloss = self.train_on_batch(batch_xs, batch_ys)
 
                    losses.append(vloss)
 
                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, mean_loss=np.mean(losses[-200:], axis=0))
                        )
                        print('Test results', self.sess.run([self.loss, self.accuracy],
                                                            feed_dict={self.x: mnist.test.images[:1000],
                                                                       self.y_target: mnist.test.labels[:1000],
                                                                       self.keep_prob: 1.0}))

 
            except KeyboardInterrupt:
                print('Stopping training!')
                pass
 
            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                self.y_target: mnist.test.labels, self.keep_prob: 1.0}))
 
 
if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()


