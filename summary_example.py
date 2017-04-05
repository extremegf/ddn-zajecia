'''
Here is a very simple example of using summaries.
Run this example.
Then run   tensorboard --logdir=out/exp_log
or         tensorboard --logdir=out   (in this case tensorboard will allow you to choose other experiments in out directory)
Then open your browser with the address: 127.0.1.1:6006 and explore

WARNING: running this code multiple time with the same LOG_DIR each time will produce strange results,
change LOG_DIR for each new run.

'''

import tensorflow as tf

LOG_DIR = 'out/exp_log'

a = tf.placeholder(dtype=tf.float32, name='a')
b = tf.placeholder(dtype=tf.float32, name='b')
sum = a + b
diff = a - b
tf.summary.scalar('sum', sum)
tf.summary.scalar('diff', diff)

all_summaries = merged = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    for i in xrange(30):
        feed_dict = {a: float(i), b: float(i)}

        sum_, diff_, summary_ = sess.run([sum, diff, all_summaries], feed_dict)
        print sum_, diff_
        summary_writer.add_summary(summary_, i)
        summary_writer.flush()

