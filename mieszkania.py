import numpy as np
import csv

def msle(ys, ps):
    assert len(ys) == len(ps)
    ys = np.array(ys)
    ps = np.array(ps)
    return np.sqrt(np.average((np.log(1+ys)-np.log(1+ps))**2))

def open_dataset(name):
    with open(name, 'rb') as csvfile:
        csvfile.readline()
        xs = []
        ys = []
        for l in csvfile:
            parts = l.strip().split(',')
            dz = parts[1]
            del parts[1]
            parts = map(float, parts)
            ys.append(parts[-1])
            del parts[-1]
            z = np.zeros(len(dzielnice))
            z[dzielnice==dz] = 1.
            xs.append(np.concatenate([np.array(parts), z]))
        xs = np.array(xs)
        ys = np.array(ys)
    print 'm2, syp, laz, rok, garaz, dzielnica[4]'
    print map(int, xs[0])
    return xs,  ys


def csv2np(col, t):
    with open('mieszkania.csv', 'rb') as csvfile:
        dr = csv.DictReader(csvfile)
        l = []
        for row in dr:
            l.append(t(row[col]))
    return np.array(l)

dzielnice = np.unique(csv2np('dzielnica', str))
