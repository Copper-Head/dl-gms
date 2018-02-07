import os
import configparser
from functools import partial

import h5py
import tensorflow as tf
import numpy as np

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
import keras.backend as K

# Config shared with docker script.
config = configparser.ConfigParser()
config.read('config.ini')
DATA_DIR = config['paths']['data_dir']
TB_LOGDIR = config['paths']['tb_logdir']
RUN_NAME = "vae/" + config['run']['name']

img_dim = 64
img_channels = 3
img_shape = (img_dim, img_dim, img_channels)
original_dim = np.product(img_shape)
intermediate_dim = 64
latent_dim = 2

BATCH_SIZE = 9
N_EPOCHS = 1000
STEPS_PER_EPOCH = 1
LEARNING_RATE = 1e-4
MAX_IMAGES = 9

# for the reparametrization trick.
epsilon_std = 1.


def sampling(args):
    """Reparametrization trick"""
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# Model definition
x = Input(shape=img_shape, name='x')
x_flat = Flatten()(x)

with tf.variable_scope('encoder'):
    h = Dense(intermediate_dim, activation='relu', name='x_to_h')(x_flat)
    z_mean = Dense(latent_dim, use_bias=False, name='h_to_mu')(h)
    z_log_var = Dense(latent_dim, use_bias=False, name='h_to_sigma')(h)

z = Lambda(sampling, name='sampling')([z_mean, z_log_var])

with tf.variable_scope('decoder'):
    h_decoder = Dense(intermediate_dim, activation='relu', use_bias=False, name='z_to_h')
    x_decoder = Dense(original_dim, activation='sigmoid', name='h_to_xhat')
    h_decoded = h_decoder(z)
    x_flat_decoded = x_decoder(h_decoded)

x_reshape = Reshape(img_shape, name='restore_img_shape')
x_decoded = x_reshape(x_flat_decoded)

# Losses & Optimization
with tf.name_scope('kl_loss'):
    kl_loss = tf.reduce_mean(
        -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1),
        name='kl_loss')
with tf.name_scope('rec_loss'):
    rec_loss = tf.reduce_mean(tf.reduce_mean((x - x_decoded)**2, axis=-1), name="rec_loss")
with tf.name_scope('vae_loss'):
    vae_loss = kl_loss + rec_loss

train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(vae_loss)

saver = tf.train.Saver(max_to_keep=None)
tb_path = os.path.join(TB_LOGDIR, RUN_NAME)
weights_path = os.path.join(config['paths']['weights_dir'], RUN_NAME)
os.makedirs(weights_path, exist_ok=True)
with tf.Session() as s, \
    tf.summary.FileWriter(tb_path, s.graph) as main_writer, \
    h5py.File(os.path.join(DATA_DIR, 'legos.h5')) as data_f:

    # Data
    data = tf.data.Dataset.from_tensor_slices(data_f['images'][:1])
    iterator = data.map(lambda im: im / 255).batch(BATCH_SIZE).repeat().make_one_shot_iterator()
    next_batch = iterator.get_next()

    # TB Summaries
    kl_summary = tf.summary.scalar('vae_kl', kl_loss)
    rec_summary = tf.summary.scalar('vae_rec', rec_loss)

    loss_summary = tf.summary.scalar('vae_global', vae_loss)
    z_dist = tf.summary.histogram('z', z)
    main_summary = tf.summary.merge([loss_summary, kl_summary, rec_summary, z_dist])

    input_img = tf.summary.image('input_image', x, max_outputs=MAX_IMAGES)
    decoded_img = tf.summary.image('decoded_image', x_decoded, max_outputs=MAX_IMAGES)
    img_summary = tf.summary.merge([input_img, decoded_img])

    # Training
    s.run(tf.global_variables_initializer())
    min_train_loss = float('inf')

    for epoch in range(N_EPOCHS):
        losses = []
        for btch in range(STEPS_PER_EPOCH):
            batch = s.run(next_batch)
            batch_loss, _, summ = s.run([vae_loss, train_step, main_summary], feed_dict={x: batch})
            losses.append(batch_loss)

        print("done training epoch: {}".format(epoch))
        main_writer.add_summary(summ, epoch)

        epoch_loss = np.mean(losses)
        if epoch_loss < min_train_loss:
            print('Saving model params!')
            saver.save(s, os.path.join(weights_path, '{:03d}--{:f}'.format(epoch, epoch_loss)))
            min_train_loss = epoch_loss
        # generate images for last epoch
        if epoch + 1 == N_EPOCHS:
            images = s.run(img_summary, feed_dict={x: batch})
            main_writer.add_summary(images)
