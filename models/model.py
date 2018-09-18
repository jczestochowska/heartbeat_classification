import numpy as np
import tensorflow as tf

from scipy.io.wavfile import read

wav = read('/home/justyna/WORKSPACE/heartbeat_classification/data/set_b/extrastole__128_1306344005749_A.wav')
x = wav[1]
x = list(map(float, x))
x = np.array([x])
x = np.transpose(x)
input_size = len(x)

dtype=tf.float64
x = tf.placeholder(shape=(input_size , 1), dtype=dtype)

Wz = tf.Variable(tf.random_normal(shape=(400, input_size), dtype=dtype))
Wr = tf.Variable(tf.random_normal(shape=(400, input_size), dtype=dtype))
Wh = tf.Variable(tf.random_normal(shape=(400, input_size), dtype=dtype))

Uz = tf.Variable(tf.random_normal(shape=(400, 400), dtype=dtype))
Ur = tf.Variable(tf.random_normal(shape=(400, 400), dtype=dtype))
Uh = tf.Variable(tf.random_normal(shape=(400, 400), dtype=dtype))

bz = tf.Variable(tf.random_normal(shape=(400, 1), dtype=dtype))
br = tf.Variable(tf.random_normal(shape=(400, 1), dtype=dtype))
bh = tf.Variable(tf.random_normal(shape=(400, 1), dtype=dtype))

hidden_state = tf.zeros(shape=(400, 1), dtype=dtype)

zt = tf.sigmoid(tf.add(tf.add(tf.matmul(Wz, x), tf.matmul(Uz, hidden_state)), bz))

rt = tf.sigmoid(tf.add(tf.add(tf.matmul(Wr, x), tf.matmul(Ur, hidden_state)), bz))

ht_hat = tf.tanh(tf.multiply(tf.add(tf.matmul(Wh, x), tf.matmul(Uh, hidden_state)),
                        tf.add(rt, bh)))

ht = tf.add(tf.multiply((1 - zt), hidden_state), tf.multiply(zt, ht_hat))

