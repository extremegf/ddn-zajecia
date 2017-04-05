import datetime

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

''''
Tasks:
1. Train a simple linear model(no hidden layers) on the mnist dataset.
   Use softmax layer and cross entropy loss i.e.
   if P is the vector of predictions, and y is one-hot encoded correct label
   the loss = \sum{i=0..n} y_i * -log(P_i)
   Train it using some variant of stochastic gradient descent.
   What performance(loss, accuracy) do you get?

2. Then change this solution to implement a Multi Layer Perceptron.
   Choose the activation function and initialization of weights.
   Make sure it is easy to change number of hidden layers, and sizes of each layer.
   What performance(loss, accuracy) do you get?

3. If you used built in tensorflow optimizer like tf.train.GradientDescentOptimizer, try
   implementing it on your own. (hint: tf.gradients method)

4. Add summaries and make sure you can see the the progress of learning in TensorBoard.
   (You can read more here https://www.tensorflow.org/get_started/summaries_and_tensorboard,
   example there is a little involved, you can look at summary_example.py for a shorter one)

5. Add periodic evaluation on the validation set (mnist.validation).
   Can you make the model overfit?

   Make sure the statistics from training and validation end up in TensorBoard
   and you can see them both on one plot.

6. Enable saving the weights of the trained model.
   Make it also possible to read the trained model.
   (You can read about saving and restoring variable in https://www.tensorflow.org/programmers_guide/variables)

Extra task:
* Show the images from the test set that the model gets wrong. Do it using TensorBoard.
* Try running your model on CIFAR-10 dataset. See what results you can get. In the future we will try
to solve this dataset with Convolutional Neural Network.
'''
class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        opt = self.optimizer # normal
        # opt = self.sgd_opt # manual sgd
        return self.sess.run([opt, self.loss, self.accuracy, self.all_summaries],
                             feed_dict={self.x: batch_xs, self.y_target: batch_ys})[1:]

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 10])

        self.vars = []
        lsizes = [784, 50, 10]
        y = self.x
        for i, (in_size, out_size) in enumerate(zip(lsizes[:-1], lsizes[1:])):
            W = tf.Variable(np.random.normal(loc=0., scale=2./ (in_size + out_size),
                                             size=[in_size, out_size]), dtype=tf.float32,
                            name="W_hidden_%d" % i)
            tf.summary.histogram(W.name, W)
            b = tf.Variable(np.zeros([out_size]), dtype=tf.float32,
                            name="b_hidden_%d" % i)
            tf.summary.histogram(b.name, b)
            self.vars += [W, b]
            y = tf.matmul(y, W) + b
            if out_size != lsizes[-1]:
                y = tf.maximum(y / 5, y)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.y_target),
                                   name='loss')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(self.y_target, 1)),
                                               tf.float32),
                                       name='accuracy')
        tf.summary.scalar(self.loss.name, self.loss)
        tf.summary.scalar(self.accuracy.name, self.accuracy)

        self.all_summaries = tf.summary.merge_all()


        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)


        self.gradients = tf.gradients(self.loss, self.vars)
        assigns = []
        for var, grad in zip(self.vars, self.gradients):
            assigns.append(tf.assign(var, var - 0.01 * grad))
        self.sgd_opt = tf.group(*assigns)


    def train(self):
        self.create_model()
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        with tf.Session() as self.sess:
            summary_writer = tf.summary.FileWriter('/tmp/tb/%s' % datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                   self.sess.graph)
            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)

                    loss, accuracy, summary = self.train_on_batch(batch_xs, batch_ys)
                    summary_writer.add_summary(summary, batch_idx)
                    summary_writer.flush()


                    losses.append(loss)

                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: loss {loss}, mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, loss=loss, mean_loss=np.mean(losses[-200:]))
                        )


            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                self.y_target: mnist.test.labels}))


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()

