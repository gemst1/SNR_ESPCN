import numpy as np
import tensorflow as tf
from glob import glob
import scipy.misc
import os

def imread(path, grayscale = True):
    if (grayscale):
        return scipy.misc.imread(path, flatten = False).astype(np.float32)
    else:
        return scipy.misc.imread(path).astype(np.float32)

def get_image(image_path, input_height=256, input_width=256,
              resize_height=256, resize_width=256,
              crop=False, grayscale=True):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def transform(image, input_height, input_width,
              resize_height=256, resize_width=256, crop=True):
  cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

# Batch normalization class
class batch_norm(object):
    def __init__(self, epsilon=1e-8, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, batch = True, train=True):
        if batch:
            return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum,
                                                updates_collections=None,
                                                epsilon=self.epsilon,
                                                scale=True,
                                                is_training=train,
                                                scope=self.name)
        else:
            return x

# fully connected layer
def linear(input, output_dim, scope=None):
    input_dim = input.get_shape()[1]
    norm = tf.random_normal_initializer(stddev=0.02)
    # xavier = tf.contrib.layers.xavier_initializers()
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input_dim, output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

# convolutional layer
def conv(filter_size, input, outputdim, strides=None, scope = None, stddev=None):
     norm = tf.random_normal_initializer(stddev=stddev)
     const = tf.constant_initializer(0.0)
     with tf.variable_scope(scope or 'fsconv'):
         w = tf.get_variable('w', [filter_size[0], filter_size[1], input.get_shape()[-1], outputdim], initializer=norm)
         b = tf.get_variable('b', [outputdim], initializer=const)
         conv = tf.nn.conv2d(input, w, strides=strides, padding='SAME')
         conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
         return conv

def SNR_ESPCN(i_sn, batch = True):
    filter_size1 = [5, 5]
    filter_size2 = [3, 3]
    filter_size3 = [3, 3]
    filter_size4 = [3, 3]
    stddev = 0.02

    h1 = tf.tanh(conv(filter_size1, i_sn, 64, strides=[1, 1, 1, 1], scope='g_h1', stddev=stddev))

    gbn_2 = batch_norm(name='gbn_2')
    h2 = tf.tanh(gbn_2(conv(filter_size2, h1, 32, strides=[1, 1, 1, 1], scope='g_h2', stddev=stddev), batch=batch))

    h3 = tf.tanh(conv(filter_size3, h2, 4, strides=[1, 1, 1, 1], scope='g_h3', stddev=stddev))
    h3 = tf.depth_to_space(h3, 2)

    h4 = (conv(filter_size4, h3, 1, strides=[1, 2, 2, 1], scope='g_h4', stddev=stddev))

    return h4

# optimizer
def optimizer(loss, var_list, learning_rate, beta1):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=var_list)
    return optimizer

def input_sample(batch_size, data):
    batch_size = batch_size//6
    input = np.random.choice(data[:1000], batch_size)
    input = np.concatenate([input, np.random.choice(data[1000:2000], batch_size)])
    input = np.concatenate([input, np.random.choice(data[2000:3000], batch_size)])
    input = np.concatenate([input, np.random.choice(data[3000:4000], batch_size)])
    input = np.concatenate([input, np.random.choice(data[4000:5000], batch_size)])
    input = np.concatenate([input, np.random.choice(data[5000:6000], batch_size)])
    return input

def real_sample(batch_size, data):
    batch_size = batch_size // 6
    input = np.random.choice(data[:1], batch_size)
    input = np.concatenate([input, np.random.choice(data[1:2], batch_size)])
    input = np.concatenate([input, np.random.choice(data[2:3], batch_size)])
    input = np.concatenate([input, np.random.choice(data[3:4], batch_size)])
    input = np.concatenate([input, np.random.choice(data[4:5], batch_size)])
    input = np.concatenate([input, np.random.choice(data[5:6], batch_size)])
    return input

class SNR(object):
    def __init__(self, num_steps, batch_size, log_every):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.input_fname = '*.jpg'

        # Train data
        self.data_train = np.concatenate([glob(os.path.join("./images_256/train/train_a", self.input_fname)),
                                          glob(os.path.join("./images_256/train/train_b", self.input_fname)),
                                          glob(os.path.join("./images_256/train/train_c", self.input_fname)),
                                          glob(os.path.join("./images_256/train/train_d", self.input_fname)),
                                          glob(os.path.join("./images_256/train/train_e", self.input_fname)),
                                          glob(os.path.join("./images_256/train/train_f", self.input_fname))])
        # Real Data
        self.data_real = np.concatenate([glob(os.path.join("./images_256/real/real_a", self.input_fname)),
                                         glob(os.path.join("./images_256/real/real_b", self.input_fname)),
                                         glob(os.path.join("./images_256/real/real_c", self.input_fname)),
                                         glob(os.path.join("./images_256/real/real_d", self.input_fname)),
                                         glob(os.path.join("./images_256/real/real_e", self.input_fname)),
                                         glob(os.path.join("./images_256/real/real_f", self.input_fname))])
        # Test Data
        self.data_test_1 = glob(os.path.join("./images_256/test/test_h", self.input_fname))
        self.data_test_2 = glob(os.path.join("./images_256/test/test_i", self.input_fname))
        self.data_test_3 = glob(os.path.join("./images_256/test/test_j", self.input_fname))
        self.data_test_4 = glob(os.path.join("./images_256/test/test_k", self.input_fname))
        self.data_test_5 = glob(os.path.join("./images_256/test/test_a", self.input_fname))

        self.data_test_1 = np.reshape([get_image(self.data_test_1[0])], [-1, 256, 256, 1])
        self.data_test_2 = np.reshape([get_image(self.data_test_2[0])], [-1, 256, 256, 1])
        self.data_test_3 = np.reshape([get_image(self.data_test_3[0])], [-1, 256, 256, 1])
        self.data_test_4 = np.reshape([get_image(self.data_test_4[0])], [-1, 256, 256, 1])
        self.data_test_5 = np.reshape([get_image(self.data_test_5[0])], [-1, 256, 256, 1])

        # learning rate
        self.learning_rate_1 = 0.0001
        self.learning_rate_2 = 0.00001
        self.beta1 = 0.5

        # image size
        self.output_size = 256

        self._create_model()

    def _create_model(self, scope=None):

        self.i_sn = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, 1])
        self.i_sn_ = tf.placeholder(tf.float32, [1, self.output_size, self.output_size, 1])
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, 1])
        self.dropout_conv = tf.placeholder(tf.float32)

        with tf.variable_scope('SNR') as scope:
            self.G = SNR_ESPCN(self.i_sn, batch=True)
            scope.reuse_variables()
            self.G_ = SNR_ESPCN(self.i_sn_, batch=True)

        # Loss Functions
        self.MSE = tf.reduce_mean(tf.squared_difference(self.x, self.G))

        self.CoC = -tf.log(1 + tf.reduce_sum(
            (self.x - tf.reshape(tf.reshape(tf.ones_like(self.x), [self.batch_size, 256 * 256]) * tf.reduce_mean(
                tf.reduce_mean(self.x, 1), 1), [self.batch_size, 256, 256, 1])) * (
                self.G - tf.reshape(tf.reshape(tf.ones_like(self.G), [self.batch_size, 256 * 256]) * tf.reduce_mean(
                    tf.reduce_mean(self.G, 1), 1), [self.batch_size, 256, 256, 1]))) / tf.sqrt(
            tf.reduce_sum(tf.square(self.x - tf.reshape(tf.reshape(
                tf.ones_like(self.x), [self.batch_size, 256 * 256]) * tf.reduce_mean(tf.reduce_mean(self.x, 1), 1),
                                                        [self.batch_size, 256, 256, 1]))) * tf.reduce_sum(
                tf.square(self.G - tf.reshape(
                    tf.reshape(tf.ones_like(self.G), [self.batch_size, 256 * 256]) * tf.reduce_mean(
                        tf.reduce_mean(self.G, 1), 1), [self.batch_size, 256, 256, 1])))))

        self.var = tf.reduce_mean(tf.image.total_variation(self.G))

        self.loss_g = self.CoC + self.MSE + 0.000001 * self.var

        # Trainable Variables
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SNR')

        # Optimizer
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate_1, self.beta1)
        self.opt_g2 = optimizer(self.loss_g, self.g_params, self.learning_rate_2, self.beta1)

    def train(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if not os.path.exists('Results/SNR'):
                os.makedirs('Results/SNR/')

            i = 0

            for step in range(self.num_steps):
                # set random seed as step
                np.random.seed(step)

                # update network
                i_sn = input_sample(self.batch_size, self.data_train)
                i_sn_sample = [get_image(sample_file) for sample_file in i_sn]
                i_sn_sample = np.reshape(i_sn_sample,[-1, 256, 256, 1])
                x = real_sample(self.batch_size, self.data_real)
                x_sample = [get_image(sample_file) for sample_file in x]
                x_sample = np.reshape(x_sample, [-1, 256, 256, 1])

                loss_g, _, MSE, CoC = sess.run([self.loss_g, self.opt_g, self.MSE, self.CoC], {self.i_sn: i_sn_sample, self.x: x_sample})

                if step % self.log_every == 0:

                    print('{}: G: {:0.6f}\tP: {:0.6f}\tC: {:0.6f}'.format(step, loss_g, MSE, CoC))

                    if step % (self.log_every*10) == 0:
                        # Sampling test Images
                        for j in range(5):
                            image_sample = self._samples(sess, j)
                            scipy.misc.toimage(image_sample, cmin=-1, cmax=1).save('Results/SNR/{}-{}.png'.format(str(j), str(i).zfill(3)))

                        i += 1

            for step in range(self.num_steps):
                # set random seed as step
                np.random.seed(step)

                # update network
                i_sn = input_sample(self.batch_size, self.data_train)
                i_sn_sample = [get_image(sample_file) for sample_file in i_sn]
                i_sn_sample = np.reshape(i_sn_sample,[-1, 256, 256, 1])
                x = real_sample(self.batch_size, self.data_real)
                x_sample = [get_image(sample_file) for sample_file in x]
                x_sample = np.reshape(x_sample, [-1, 256, 256, 1])

                loss_g, _, MSE, CoC = sess.run([self.loss_g, self.opt_g, self.MSE, self.CoC], {self.i_sn: i_sn_sample, self.x: x_sample})

                if step % self.log_every == 0:

                    print('{}: G: {:0.6f}\tP: {:0.6f}\tC: {:0.6f}'.format(step, loss_g, MSE, CoC))

                    if step % (self.log_every*10) == 0:
                        # Sampling test Images
                        for j in range(5):
                            image_sample = self._samples(sess, j)
                            scipy.misc.toimage(image_sample, cmin=-1, cmax=1).save('Results/SNR/{}-{}.png'.format(str(j), str(i).zfill(3)))

                        i += 1

    def _samples(self, sess, j):

        if j == 0:
            test_image = self.data_test_1
        elif j == 1:
            test_image = self.data_test_2
        elif j == 2:
            test_image = self.data_test_3
        elif j == 3:
            test_image = self.data_test_4
        else:
            test_image = self.data_test_5

        gen_image = sess.run(self.G_, feed_dict={self.i_sn_: test_image, self.dropout_conv: 1})
        gen_image = np.reshape(gen_image, [256, 256])

        return gen_image

def main():
    # with tf.device('/cpu:0'):
    model = SNR(
        2500,  # training iteration steps
        6,  # batch size per data set
        10,  # log step
    )
    model.train()

if __name__ == '__main__':
    main()