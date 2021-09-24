import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Layer, BatchNormalization, Conv2D, Conv2DTranspose, ReLU


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder(Nt,Nd,latent_dim=64):

    capa_in = Input(shape=(Nt,Nd,1))

    x = Conv2D(8, (3,3), strides=(2,2), padding='same')(capa_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(16, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(32, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Flatten()(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    return Model(inputs=capa_in, outputs=[z_mean,z_log_var,z])


def create_decoder(Nt,Nd,latent_dim=64):

    latent_in = Input(shape=(latent_dim,))

    x = Dense(Nt/8 * Nd/8 * 32,)(latent_in)
    x = Reshape((Nt//8, Nd//8, 32))(x)

    x = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(8, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x_mean = Conv2DTranspose(1, (2,2), activation='linear', padding='same')(x)
    x_log_var = Conv2DTranspose(1, (2,2), activation='linear', padding='same')(x)

    return Model(inputs=latent_in, outputs=[x_mean,x_log_var])


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.loss_kl_tracker = Mean()
        self.loss_rec_tracker = Mean()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.loss_kl_tracker,
            self.loss_rec_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            x_mean, x_log_var = self.decoder(z)
            loss_kl = K.mean(K.sum(-0.5*(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean)),axis=1))
            loss_rec = K.mean(K.sum(0.5*(K.log(2*np.pi)+x_log_var+(x-x_mean)**2/K.exp(x_log_var)),axis=(1,2)))
            total_loss = loss_kl + loss_rec
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.loss_kl_tracker.update_state(loss_kl)
        self.loss_rec_tracker.update_state(loss_rec)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "loss_kl": self.loss_kl_tracker.result(),
            "loss_rec": self.loss_rec_tracker.result(),
        }
