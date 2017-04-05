import numpy as np
import os

IMG_DEPTH = 3
IMG_SIDE = 32
IMG_PIX = IMG_SIDE ** 2 * IMG_DEPTH

def _unpickle_cifar10(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class Cifar10Dataset(object):
    def __init__(self, files):
        labels = []
        data = []
        for f in files:
            raw_data = _unpickle_cifar10(f)
            labels.append(raw_data['labels'])
            data.append(raw_data['data'])

        self.images = np.concatenate(data).reshape((-1, 3, 32, 32))
        self.images = np.transpose(self.images, axes=(0, 2, 3, 1))
        self.images = np.reshape(self.images, (-1, IMG_PIX))
        labels = np.concatenate(labels)

        # 1-hot
        n = self.n = self.images.shape[0]
        self.labels = np.zeros((n, 10))
        self.labels[np.arange(n), labels] = 1

        self.batch_permut = np.random.permutation(n)
        self.batch_pos = 0

    def next_batch(self, batch_size):
        if batch_size + self.batch_pos >= self.n:
            self.batch_permut = np.random.permutation(self.n)
            self.batch_pos = 0
        rng = self.batch_permut[self.batch_pos:self.batch_pos+batch_size]
        self.batch_pos += batch_size
        return self.images[rng], self.labels[rng]


train = Cifar10Dataset([os.path.join("cifar-10-batches-py", "data_batch_%d" % i)
                        for i in [1, 2, 3, 4]])
validation = Cifar10Dataset([os.path.join("cifar-10-batches-py", "data_batch_%d" % i)
                        for i in [5]])
test = Cifar10Dataset([os.path.join("cifar-10-batches-py", "test_batch")])


