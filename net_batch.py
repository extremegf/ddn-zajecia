import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

from Cryptodome.Random import random

IMPORT_TIME = datetime.datetime.now()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid
    return sigmoid(z)*(1-sigmoid(z))

class Layer(object):
    def get_weigths(self):
        return []

    def get_gradients(self):
        return []

    def is_bias(self):
        return []

class FC(Layer):
    def __init__(self, ins, outs):
        self.w = np.random.normal(scale=(1.0  / (ins + outs)), size=[ins, outs])
        self.b = np.random.normal(scale=0.1, size=[outs])

    def forward(self, input):
        # input.shape = (batch_size, inp_width)
        # w.shape = (inp_width, outp_width)
        # b.shape = (outp_width)
        return sigmoid(np.dot(input, self.w) + self.b)

    def backward(self, orig_input, orig_output, output_gradient):
        # gp ~ bs x in_w
        # g ~ bs x out_w
        # Lg ~ bs x out_w
        gp = orig_input
        g = orig_output
        Lg = output_gradient
        Lf = Lg * g * (1.0 - g)

        # gp.T ~ in_w x bs
        # Lf = bs x out_w
        #
        Lw = np.dot(gp.T, Lf)
        self.grad_w = Lw
        self.grad_b = np.sum(Lf.T, axis=1)

        # Lf ~ bs x out_w
        # w.T ~ out_w x in_w
        Lgp = np.dot(Lf, self.w.T)
        return Lgp

    def get_gradients(self):
        return [self.grad_w, self.grad_b]

    def get_weigths(self):
        return [self.w, self.b]

    def is_bias(self):
        return [False, True]

class SepLogRegLoss(Layer):
    def setup(self, exp_input):
        self.exp_input = exp_input

    def forward(self, input):
        # g ~ bs x cls
        # y ~ bs x cls
        g = input
        y = self.exp_input

        sep_losses = -y*np.log(g) - (1.-y)*np.log(1.-g)
        per_case_loss = np.sum(sep_losses, axis=1)
        loss = np.average(per_case_loss)
        return loss

    def backward(self, orig_input, orig_output, output_gradient):
        bs = orig_input.shape[0]
        y = self.exp_input
        g = orig_input
        return -1./bs * (y / g - (1.-y) / (1.-g))


class Net(object):
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer


    def forward_backward(self, input):
        inp = input
        inouts = []
        for l in self.layers:
            out = l.forward(inp)
            inouts.append((inp, out))
            inp = out

        grad = 1.
        for l, (oinp, oout) in reversed(zip(self.layers, inouts)):
            grad = l.backward(oinp, oout, grad)

        return inouts

    def grad_descent(self):
        ws = []
        dws = []
        is_bias = []
        for l in self.layers:
            ws += l.get_weigths()
            dws += l.get_gradients()
            is_bias += l.is_bias()
        self.optimizer.run(ws, dws, is_bias)


class Network(object):
    def __init__(self, sizes, dropout, optimizer):
        assert len(dropout) == len(sizes) - 1
        self.losses = []
        self.dropout_layers = []

        layers = []
        for ins, outs, do_prob in zip(sizes[:-1], sizes[1:], dropout):
            do = DropOut(do_prob)
            self.dropout_layers += [do]
            layers += [FC(ins, outs)]
            if do_prob > 0.:
                layers += [do]
        self.loss_layer = SepLogRegLoss()
        layers += [self.loss_layer]
        self.net = Net(layers, optimizer)

    def feedforward(self, a):
        return a

    def run_minibach(self, mini_batch, is_training=True):
        xs = mini_batch[0].reshape(-1, 784)
        ys = mini_batch[1].reshape(-1, 10)
        assert xs.shape[0] == ys.shape[0]
        self.loss_layer.setup(ys)
        for do in self.dropout_layers:
            do.set_mode(is_training)
        return self.net.forward_backward(xs)

    def numeric_gradient_check(self, mini_batch, eps=1e-6):
        results = self.run_minibach(mini_batch)
        base_loss = results[-1][1]

        print 'Base loss', base_loss
        print 'Changing single weigth'
        l=1
        wt=1
        ind = 9
        lw = self.net.layers[l].get_weigths()[wt]
        grad_lw = self.net.layers[l].get_gradients()[wt]
        org_grad = grad_lw.flat[ind]
        orig = lw.flat[ind]
        lw.flat[ind] += eps
        results = self.run_minibach(mini_batch)
        lw0_loss = results[-1][1]
        print 'lw[ind] change loss', base_loss, 'delta', base_loss - lw0_loss
        ngrad = (lw0_loss - base_loss) / eps
        print 'lw[ind] grad', ngrad, 'analitical grad', org_grad
        lw.flat[ind] = orig

        results = self.run_minibach(mini_batch)
        print 'After revert', results[-1][1]

        fig, axs = plt.subplots(len(self.net.layers), 2, figsize=(10, 10))
        diffs = []
        for row, l in enumerate(self.net.layers):
            for col, w in enumerate(l.get_gradients()):
                for i in xrange(len(w.flat)):
                    diffs.append(w.flat[i])
                axs[row,col].hist(list(w.flat), 50)

        num_diffs = []
        for l in self.net.layers:
            for w in l.get_weigths():
                for i in xrange(len(w.flat)):
                    orig = w.flat[i]
                    w.flat[i] += eps
                    results = self.run_minibach(mini_batch)
                    new_loss = results[-1][1]
                    num_diff = (new_loss - base_loss) / eps
                    num_diffs.append(num_diff)
                    w.flat[i] = orig

        num_diffs = np.array(num_diffs)
        diffs = np.array(diffs)
        error = np.abs(num_diffs-diffs)
        print np.min(error), np.average(error), np.max(error)

    def update_mini_batch(self, mini_batch):
        outputs = self.run_minibach(mini_batch)
        loss = outputs[-1][1]
        self.losses.append(loss)
        self.net.grad_descent()
        return loss

    def evaluate(self, test_data):
        # Count the number of correct answers for test_data
        outputs = self.run_minibach(test_data, is_training=False)
        net_output = outputs[-1][0]
        test_results = [(np.argmax(net_output[i]),
                         np.argmax(test_data[1][i]))
                        for i in range(len(test_data[0]))]
        #print test_results
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, test_data=None):
        self.losses = []
        train_size = training_data.images.shape[0]
        if test_data:
            test_size = test_data.images.shape[0]
        for j in xrange(epochs):
            for k in range(train_size/mini_batch_size):
                self.update_mini_batch(training_data.next_batch(mini_batch_size))
            if test_data and (j % 10 == 0 or j == epochs-1):
                res = np.mean([self.evaluate(test_data.next_batch(mini_batch_size)) for k in range(test_size/mini_batch_size)])/mini_batch_size
                print "Epoch {0}: {1}".format(j, res)
        print 'Time since import:', datetime.datetime.now()-IMPORT_TIME
        plt.plot(self.losses)


class Optimizer(object):
    def run(self, ws, dws, is_bias):
        pass


class TrivialGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def run(self, ws, dws, is_bias):
        for w, dw in zip(ws, dws):
            w -= self.lr * dw


class Momentum(Optimizer):
    def __init__(self, mem=0.9, rate=0.001):
        self.ms = None
        self.mem = mem
        self.rate = rate

    def run(self, ws, dws, is_bias):
        if self.ms is None:
            self.ms = [np.zeros(w.shape) for w in ws]

        for m, w, dw in zip(self.ms, ws, dws):
            m *= self.mem;
            m -= (1. - self.mem) * self.rate * dw
            w += m


class NesterovMomentum(Optimizer):
    def __init__(self, mem=0.9, rate=0.001):
        self.ms = None
        self.mem = mem
        self.rate = rate

    def run(self, ws, dws, is_bias):
        if self.ms is None:
            self.ms = [np.zeros(w.shape) for w in ws]

        for m, w, dw in zip(self.ms, ws, dws):
            w -= (1. - self.mem) * self.rate * dw
            m *= self.mem
            m -= (1. - self.mem) * self.rate * dw
            w += m * self.mem


class RMSProp(Optimizer):
    def __init__(self, mem=0.9, rate=0.001):
        self.Gs = None
        self.mem = mem
        self.rate = rate

    def run(self, ws, dws, is_bias):
        if self.Gs is None:
            self.Gs = [np.zeros(w.shape) for w in ws]

        for G, w, dw in zip(self.Gs, ws, dws):
            G *= self.mem
            G += (1.-self.mem) * (dw**2)
            w -= self.rate / np.sqrt(G + 1e-8) * dw


class DropOut(Layer):
    def __init__(self, do_prob):
        self.do_prob = do_prob
        self.is_training = True

    def set_mode(self, is_training):
        self.is_training = is_training

    def forward(self, input):
        if self.is_training:
            self.mask = (np.random.uniform(0, 1, size=input.shape[1:]) > self.do_prob).astype(input.dtype)
            return self.mask * input
        else:
            return (1. - self.do_prob) * input

    def backward(self, orig_input, orig_output, output_gradient):
        if self.is_training:
            return self.mask * output_gradient
        else:
            return (1. - self.do_prob) * output_gradient


class RMSPropVarWatch(Optimizer):
    def __init__(self, watch_cnt=1, mem=0.9, rate=0.001):
        self.Gs = None
        self.mem = mem
        self.rate = rate
        self.watched = None
        self.plot_data = []

    def run(self, ws, dws, is_bias):
        if self.Gs is None:
            self.Gs = [np.zeros(w.shape) for w in ws]

        if self.watched is None:
            self.watched = []
            self.plot_data = []
            for i, w in enumerate(ws):
                k = random.randint(0, len(w.flat))
                self.watched.append((i, k))
                self.plot_data.append(([], []))

        for i, k in self.watched:
            self.plot_data[i][0].append(ws[i].flat[k])
            self.plot_data[i][1].append(dws[i].flat[k])

        for G, w, dw in zip(self.Gs, ws, dws):
            G *= self.mem
            G += (1.-self.mem) * (dw**2)
            w -= self.rate / np.sqrt(G + 1e-8) * dw

    def plt(self):
        fig, axs = plt.subplots(len(self.watched), 1, figsize=(20, 20))
        for i, data in enumerate(self.plot_data):
            axs[i].scatter(data[0], data[1])

