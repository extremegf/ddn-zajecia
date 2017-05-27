# Assignment 1 DNN Przemyslaw Horban

import datetime
import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

TB_LOGS = '/tmp/as1/'
CHECKPOINT_TF = '/tmp/checkpoint.tf'
OPT_X_PATH = "/tmp/opt_x.npy"


class DeepMnistTrainer(object):
    def weight_variable(self, shape, stddev, name='weight'):
        initializer = tf.random_normal(shape, stddev=stddev)
        return tf.Variable(initializer, name=name)

    def bias_variable(self, shape, bias, name='bias'):
        initializer = tf.constant(bias, shape=shape)
        return tf.Variable(initializer, name=name)

    def augment_batch(self, image_batch):
        image_batch = tf.reshape(image_batch, (-1, 28, 28, 1))
        reshaped_image = tf.image.resize_image_with_crop_or_pad(image_batch, 35, 35)

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, tf.shape(image_batch))

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=0.05)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.95, upper=1.05)
        return tf.reshape(distorted_image, (-1, 28 * 28))


    def convolutional_layers(self, input, conv_depth_list=[32, 64]):
        with tf.name_scope("convolutional_layers"):
            signal = tf.reshape(input, [-1, 28, 28, 1])
            for idx, new_depth in enumerate(conv_depth_list):
                cur_depth = int(signal.get_shape()[3])
                W = self.weight_variable([5, 5, cur_depth, new_depth], 0.1, 'conv_W')
                signal = tf.nn.conv2d(signal, W, strides=[1, 1, 1, 1], padding='SAME')

                # Normal bias-non-batchnorm version
                # b = self.bias_variable([new_depth], 0.0)
                # signal += b

                # Batchnorm gamma beta
                with tf.name_scope("conv_batch_norm"):
                    neur_avg = tf.reduce_mean(signal, axis=[0, 1, 2])
                    neur_variance = tf.reduce_mean(
                        (signal - neur_avg) ** 2, axis=[0, 1, 2]
                    )
                    signal = (signal - neur_avg) / tf.sqrt(neur_variance + 1e-8);

                    beta = self.bias_variable([new_depth], 0.0)
                    gamma = self.bias_variable([new_depth], 1.0)
                    signal = signal * gamma + beta


                with tf.name_scope("leaky_relu"):
                    signal = tf.maximum(signal / 100., signal)  # leaky relu
                signal = tf.nn.max_pool(signal,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='maxpool')

                print 'Conv. layer output shape', signal.get_shape()

            sh = signal.get_shape()
            signal = tf.reshape(signal, [-1, int(sh[1] * sh[2] * sh[3])])

            return signal

    def fully_conn_layers(self, input, layer_sizes=[1024, 10]):
        with tf.name_scope("fully_conn_layers"):
            signal = input
            self.keep_prob = tf.placeholder(tf.float32)
            for idx, new_num_neurons in enumerate(layer_sizes):
                cur_num_neurons = int(signal.get_shape()[1])
                with tf.variable_scope('fc_' + str(idx + 1)):
                    W_fc = self.weight_variable([cur_num_neurons, new_num_neurons],
                                                stddev=(math.sqrt(2. / cur_num_neurons)),
                                                name='fc_W')

                signal = tf.matmul(signal, W_fc)

                if idx != len(layer_sizes) - 1:
                    with tf.name_scope('fc_batchnorm'):
                        neur_avg = tf.reduce_mean(signal, axis=0)
                        neur_variance = tf.reduce_mean(
                            (signal - neur_avg) ** 2,
                            axis=0
                        )
                        signal = (signal - neur_avg) / tf.sqrt(neur_variance + 1e-8)
                        beta = self.bias_variable([new_num_neurons], 0.0, 'beta')
                        gamma = self.bias_variable([new_num_neurons], 1.0, 'gamma')
                        signal = gamma * signal + beta

                    with tf.name_scope("leaky_relu"):
                        signal = tf.maximum(signal / 100., signal)  # leaky relu

                    signal = tf.nn.dropout(signal, self.keep_prob)

                else:  # readout layer
                    bias = self.bias_variable([new_num_neurons], 0.1, 'readout_bias')
                    signal += bias

                print 'shape', signal.get_shape()
            return signal

    def create_model(self, lr_decay=True, use_augmentation=True, use_adam=False):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')

        self.y_target = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope("model_restore_var_set") as scope:
            signal = self.x
            if use_augmentation:
                signal = self.augment_batch(signal)
            tf.summary.image("Train image", tf.reshape(signal, [-1, 28, 28, 1]), 5)
            print 'shape', signal.get_shape()
            signal = self.convolutional_layers(signal, conv_depth_list=[32, 64])
            signal = self.fully_conn_layers(signal, layer_sizes=[1024, 10])

        with tf.name_scope("loss_calculation"):
            self.y_logits = signal
            self.y_distr = tf.nn.softmax(self.y_logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y_target))
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("optimizer"):
            global_step = tf.Variable(0, trainable=False)
            if lr_decay:
                lr = tf.train.polynomial_decay(0.003, global_step, 10000, 0.00001)
            else:
                lr = tf.Variable(0.0001)
            tf.summary.scalar("learning_rate", lr)
            if use_adam:
                self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=global_step)
            else:
                self.train_step = tf.train.RMSPropOptimizer(lr).minimize(self.loss, global_step=global_step)

        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()


    def train(self, batches_n=10000, mb_size=128, load_path=None, save_path=None):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        with tf.Session() as self.sess:
            date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            summary_writer = tf.summary.FileWriter(TB_LOGS + '%s-train' % date_str, self.sess.graph)

            if load_path:
                self.load(load_path)
            else:
                tf.global_variables_initializer().run()
            try:
                loss_acc_pairs = []
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = self.mnist.train.next_batch(mb_size)
                    _train_step, _loss, _accuracy, _summary = self.sess.run(
                        [self.train_step, self.loss, self.accuracy, self.summary],
                        feed_dict={self.x: batch_xs,
                                   self.y_target: batch_ys,
                                   self.keep_prob: 0.5})
                    loss_acc_pairs.append([_loss, _accuracy])
                    summary_writer.add_summary(_summary, batch_idx)


                    if batch_idx % 100 == 0:
                        mean_loss, mean_acc = np.mean(loss_acc_pairs[-200:], axis=0)
                        print('Batch {batch_idx}: mean_loss {mean_loss} mean_accuracy {acc:.3f}%'.format(
                            batch_idx=batch_idx, mean_loss=mean_loss, acc=mean_acc * 100.)
                        )

                        if batch_idx % 1000 == 0:
                            self.test_results()


            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            self.test_results()

            if save_path:
                self.save(save_path)

    def run_model_on(self, x, load_path):
        with tf.Session() as self.sess:
            self.load(load_path)
            y_distr, y_logits = self.sess.run(
                [self.y_distr, self.y_logits],
                feed_dict={self.x: x, self.keep_prob: 1.0})
            return y_distr, y_logits

    def load(self, path):
        self.saver.restore(self.sess, path)
        print("Model loaded from file: %s" % path)

    def save(self, path):
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, path)
        print("Model saved in file: %s" % save_path)

    def test_results(self):
        test_loss, test_acc = self.sess.run([self.loss, self.accuracy],
                                            feed_dict={self.x: self.mnist.test.images,
                                                       self.y_target: self.mnist.test.labels,
                                                       self.keep_prob: 1.0})
        print('Test set results - loss: {loss} accuracy {acc:.3f}%'.format(
            loss=test_loss, acc=test_acc * 100
        ))

class BestImageMnist(DeepMnistTrainer):
    def create_model(self):
        self.x = tf.Variable(tf.random_normal([10, 784], stddev=0.001), 'opt_x')
        self.x_aug = tf.placeholder(tf.float32, [None, 784], name='x')
        signal = tf.concat([self.x, self.x_aug], axis=0)
        self.y_target_aug = tf.placeholder(tf.float32, [None, 10])
        self.y_target = tf.concat([
            tf.constant(np.eye(10, 10), dtype=self.x.dtype),
            self.y_target_aug], axis=0)

        with tf.name_scope("model_restore_var_set") as scope:
            print 'shape', signal.get_shape()
            signal = self.convolutional_layers(signal, conv_depth_list=[32, 64])
            signal = self.fully_conn_layers(signal, layer_sizes=[1024, 10])

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        with tf.name_scope("loss_calculation"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y_target))
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.image("Most engaging images", tf.reshape(self.x, [-1, 28, 28, 1]), 10)
        self.summary = tf.summary.merge_all()

        with tf.name_scope("optimizer") as scope:
            self.train_step = tf.train.RMSPropOptimizer(0.0001).minimize(self.loss, var_list=[self.x])


    def train_for_x(self, steps=10000, load_path=None):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        with tf.Session() as self.sess:
            date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            summary_writer = tf.summary.FileWriter(TB_LOGS + '%s-train-for-x' % date_str, self.sess.graph)

            tf.global_variables_initializer().run()
            self.load(load_path)
            try:
                loss_acc_pairs = []
                for step in range(steps):
                    batch_xs, batch_ys = mnist.train.next_batch(10)
                    _train_step, _loss, _accuracy, _summary = self.sess.run(
                        [self.train_step, self.loss, self.accuracy, self.summary],
                        feed_dict={
                            self.x_aug: batch_xs,
                            self.y_target_aug: batch_ys,
                            self.keep_prob: 1.0})
                    loss_acc_pairs.append([_loss, _accuracy])
                    summary_writer.add_summary(_summary, step)


                    if step % 100 == 0:
                        mean_loss, mean_acc = np.mean(loss_acc_pairs[-200:], axis=0)
                        print('Step {step}: mean_loss {mean_loss} mean_accuracy {acc:.3f}%'.format(
                            step=step, mean_loss=mean_loss, acc=mean_acc * 100.)
                        )

            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            return self.sess.run([self.x])



def train_mnist(should_load=False):
    tf.reset_default_graph()

    trainer = DeepMnistTrainer()
    trainer.create_model()
    if should_load:
        trainer.train(load_path=CHECKPOINT_TF, save_path=CHECKPOINT_TF)
    else:
        trainer.train(save_path=CHECKPOINT_TF)


def find_fooling_images(path):
    tf.reset_default_graph()

    trainer = BestImageMnist()
    trainer.create_model()
    x = trainer.train_for_x(steps=5000, load_path=CHECKPOINT_TF)
    np.save(path, x)


def demonstrate_that_we_fooled_the_network(path):
    tf.reset_default_graph()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = np.concatenate([np.load(path)[0], mnist.train.images[:10]], axis=0)

    trainer = DeepMnistTrainer()
    trainer.create_model()
    y_distr, y_logits = trainer.run_model_on(x, CHECKPOINT_TF)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    print 'y_distr:\n', y_distr
    print 'y_logits:\n', y_logits


if __name__ == '__main__':
    train_mnist(should_load=False)
    # find_fooling_images(OPT_X_PATH)
    # demonstrate_that_we_fooled_the_network(OPT_X_PATH)


