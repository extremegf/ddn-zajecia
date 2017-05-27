import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


''''
Task 1. Write simple RNN to recognize MNIST digits.
The image is 28x28. Flatten it to a 784 vector.
Pick some divisior d of 784, e.g. d = 28. 
At each timestep the input will be d bits of the image. 
Thus the sequence length will be 784 / d
You should be able to get over 93% accuracy
Write your own implementation of RNN, you can look at the one from the slide,
but do not copy it blindly.

Task 2. 
Same, but use LSTM instead of simple RNN.
What accuracy do you get.
Experiment with choosing d, compare RNN and LSTM.
Again do not use builtin Tensorflow implementation. Write your own :)

Task 3*.
Make LSTM a deep bidirectional, multilayer LSTM.
'''

class SingleLSTMLayer(object):
    def __init__(self, x, d, steps_n, hidden_n=100):
        self.d = d
        self.steps_n = steps_n
        self.hidden_n = hidden_n

        stddev=0.1
        with tf.variable_scope('lstm'):
            self.h0 = tf.Variable(name='h0', initial_value=tf.truncated_normal(shape=[hidden_n], stddev=stddev), dtype=tf.float32)
            self.c0 = tf.Variable(name='c0', initial_value=tf.truncated_normal(shape=[hidden_n], stddev=stddev), dtype=tf.float32)

            self.W = tf.Variable(name='W', initial_value=tf.truncated_normal(shape=[self.d + hidden_n, 4 * hidden_n], stddev=stddev), dtype=tf.float32)
            self.bias = tf.Variable(name='bias', initial_value=tf.truncated_normal(shape=[4 * hidden_n], stddev=stddev), dtype=tf.float32)

        self.h = {}
        self.h[-1] = tf.tile(tf.reshape(self.h0, (1, -1)), [tf.shape(x)[0], 1])
        self.c = tf.tile(tf.reshape(self.c0, (1, -1)), [tf.shape(x)[0], 1])
        for step in range(self.steps_n):
            step_x = x[:, step, :]
            x_h = tf.concat([step_x, self.h[step-1]], axis=1)
            # h shape: bs x steps x d

            ih = tf.matmul(x_h, self.W) + self.bias
            i = tf.sigmoid(ih[:, 0*hidden_n:1*hidden_n])
            f = tf.sigmoid(ih[:, 1*hidden_n:2*hidden_n])
            o = tf.sigmoid(ih[:, 2*hidden_n:3*hidden_n])
            g = tf.tanh(ih[:, 3*hidden_n:4*hidden_n])

            self.c = f * self.c + i * g
            self.h[step] = o * tf.tanh(self.c)

    def combined_h(self):
        n = max(self.h.keys())+1
        to_concat = []
        for step in range(n):
            to_concat.append(tf.reshape(self.h[step], (-1, 1, self.hidden_n)))
            tf.assert_equal(tf.shape(to_concat[-1])[0], 128)
        concat = tf.concat(to_concat, axis=1)
        tf.assert_equal(tf.shape(to_concat[-1])[1], n)
        return concat


class BiLSTMLayer(object):
    def __init__(self, x, d, steps_n, hidden_n=100):
        self.forward = SingleLSTMLayer(x, d, steps_n, hidden_n)
        rev_x = tf.reverse(x, axis=[1])
        self.reverse = SingleLSTMLayer(rev_x, d, steps_n, hidden_n)

    def combined_h(self):
        fwh = self.forward.combined_h()
        rwh = self.reverse.combined_h()

        return tf.concat([fwh, tf.reverse(rwh, axis=[1])], axis=2)

    def last_h(self):
        return self.combined_h()[:,-1,:]

class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        transf_x = np.reshape(batch_xs, (-1, self.steps_n, self.d))
        return self.sess.run([self.loss, self.accuracy, self.train_step],
                             feed_dict={self.x: transf_x,
                                        self.y_target: batch_ys})[:2]

    def create_simple_rnn_model(self):
        self.d = 28
        self.steps_n = 784 / self.d
        HSZ = 100

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.steps_n, self.d])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        stddev=0.1

        with tf.variable_scope('rnn'):
            self.h0 = tf.Variable(name='h0',
                                  initial_value=tf.truncated_normal(shape=[HSZ], stddev=stddev), dtype=tf.float32)
            self.bias = tf.Variable(name='bias',
                                    initial_value=tf.truncated_normal(shape=[HSZ], stddev=stddev), dtype=tf.float32)
            self.W = tf.Variable(name='W',
                                 initial_value=tf.truncated_normal(shape=[self.d + HSZ, HSZ], stddev=stddev), dtype=tf.float32)
            self.out_W = tf.Variable(name='out_W',
                                     initial_value=tf.truncated_normal(shape=(HSZ, 10), stddev=stddev), dtype=tf.float32)
            self.out_bias = tf.Variable(name='out_bias',
                                    initial_value=tf.truncated_normal(shape=(10,), stddev=stddev), dtype=tf.float32)

        self.h = tf.tile(tf.reshape(self.h0, (1,-1)), [tf.shape(self.x)[0], 1]) # to jest sliskie
        for step in range(self.steps_n):
            step_x = self.x[:, step, :]
            x_h = tf.concat([step_x, self.h], axis=1)
            # h shape: bs x steps x d
            self.h = tf.tanh(tf.matmul(x_h, self.W) + self.bias)

        signal = tf.matmul(self.h, self.out_W) + self.out_bias
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_target, logits=signal))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)


    def create_lstm_model(self):
        self.d = 28
        self.steps_n = 784 / self.d
        hidden_n = 100

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.steps_n, self.d])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        stddev=0.1

        with tf.variable_scope('lstm'):
            self.out_W = tf.Variable(name='out_W', initial_value=tf.truncated_normal(shape=[hidden_n, 10], stddev=stddev), dtype=tf.float32)
            self.out_bias = tf.Variable(name='out_bias', initial_value=tf.truncated_normal(shape=[10], stddev=stddev), dtype=tf.float32)
            layer = SingleLSTMLayer(self.x, self.d, self.steps_n, hidden_n=hidden_n)

        last_h = layer.h[max(layer.h.keys())]
        signal = tf.matmul(last_h, self.out_W) + self.out_bias
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_target, logits=signal))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)


    def create_bi_lstm_model(self):
        self.d = 28
        self.steps_n = 784 / self.d
        hidden_n = 100

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.steps_n, self.d])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        stddev=0.1

        with tf.variable_scope('lstm'):
            self.out_W = tf.Variable(name='out_W', initial_value=tf.truncated_normal(shape=[hidden_n*2, 10], stddev=stddev), dtype=tf.float32)
            self.out_bias = tf.Variable(name='out_bias', initial_value=tf.truncated_normal(shape=[10], stddev=stddev), dtype=tf.float32)
            layer = BiLSTMLayer(self.x, self.d, self.steps_n, hidden_n=hidden_n)

        last_h = layer.last_h()
        signal = tf.matmul(last_h, self.out_W) + self.out_bias
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_target, logits=signal))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)


    def create_deep_lstm_model(self):
        self.d = 28
        self.steps_n = 784 / self.d
        hidden_n = 100

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.steps_n, self.d])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        stddev=0.1

        with tf.variable_scope('lstm'):
            self.out_W = tf.Variable(name='out_W', initial_value=tf.truncated_normal(shape=[hidden_n, 10], stddev=stddev), dtype=tf.float32)
            self.out_bias = tf.Variable(name='out_bias', initial_value=tf.truncated_normal(shape=[10], stddev=stddev), dtype=tf.float32)
            layer = SingleLSTMLayer(self.x, self.d, self.steps_n, hidden_n=hidden_n)
            layer1 = SingleLSTMLayer(layer.combined_h(), hidden_n, self.steps_n, hidden_n=hidden_n)
            layer2 = SingleLSTMLayer(layer1.combined_h(), hidden_n, self.steps_n, hidden_n=hidden_n)

        last_h = layer2.h[max(layer2.h.keys())]
        signal = tf.matmul(last_h, self.out_W) + self.out_bias
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_target, logits=signal))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)


    def create_deep_bi_lstm_model(self):
        self.d = 28
        self.steps_n = 784 / self.d
        hidden_n = 100

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.steps_n, self.d])
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        stddev=0.1

        with tf.variable_scope('lstm'):
            self.out_W = tf.Variable(name='out_W', initial_value=tf.truncated_normal(shape=[hidden_n*2, 10], stddev=stddev), dtype=tf.float32)
            self.out_bias = tf.Variable(name='out_bias', initial_value=tf.truncated_normal(shape=[10], stddev=stddev), dtype=tf.float32)
            layer = BiLSTMLayer(self.x, self.d, self.steps_n, hidden_n=hidden_n)
            layer2 = BiLSTMLayer(layer.combined_h(), 2*hidden_n, self.steps_n, hidden_n=hidden_n)

        last_h = layer2.last_h()
        signal = tf.matmul(last_h, self.out_W) + self.out_bias
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_target, logits=signal))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def train(self):
        # self.create_deep_lstm_model()
        self.create_deep_bi_lstm_model()
        # self.create_bi_lstm_model()
        # self.create_lstm_model()
        # self.create_simple_rnn_model()
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)

                    loss, accuracy = self.train_on_batch(batch_xs, batch_ys)


                    losses.append(loss)

                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: loss {loss}, mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, loss=loss, mean_loss=np.mean(losses[-200:]))
                        )


            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            test_x, test_y = mnist.test.next_batch(2000)
            transf_x = np.reshape(test_x, (-1, self.steps_n, self.d))
            loss, acc = self.sess.run([self.loss, self.accuracy],
                                 feed_dict={self.x: transf_x,
                                            self.y_target: test_y})
            print "Test set loss {loss} accuracy {acc}".format(
                loss=loss, acc=acc)


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()

