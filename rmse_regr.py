import matplotlib.pyplot as plt
import numpy as np

class RMSERegr(object):
    def __init__(self):
        pass

    def regression(self, x, y, lrates):
        w = np.zeros(len(x[0]))

        loss_lst = []
        for lr in lrates:
            dw = 2. * (x.T * (np.sum(x*w, axis=1)-y)).T
            dw = np.average(dw, axis=0)
            w -= dw * lr
            loss_lst.append(RMSERegr.rmse(x, y, w))

        plt.plot(loss_lst)
        print 'RMSE:', RMSERegr.rmse(x, y, w)
        if len(w) < 10:
            print 'w:', w
        return w

    @staticmethod
    def rmse(x, y, w):
        assert len(y) == len(x)
        x = np.array(x)
        y = np.array(y)
        return np.sqrt(np.average((np.sum(x*w, axis=1) - y) ** 2))
