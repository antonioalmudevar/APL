import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Input, Dense, Reshape, Layer, LeakyReLU, GlobalAveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose

def create_generator(Nt,Nd):
    input_layer = Input(name='input', shape=(Nt,Nd,1))

    x = Conv2D(8, (3,3), strides=(2,2), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(16, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(32, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    y = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2DTranspose(8, (3,3), strides=(2,2), padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2DTranspose(1, (1, 1), strides=(1,1), padding='same', name='decoder_output', activation='tanh')(y)

    return Model(inputs=input_layer, outputs=y)
    
def create_feature_extractor(Nt,Nd):
    input_layer = Input(name='input', shape=(Nt,Nd,1))

    f = Conv2D(8, (3,3), strides=(2,2), padding='same')(input_layer)
    f = BatchNormalization()(f)
    f = LeakyReLU()(f)

    f = Conv2D(16, (3,3), strides=(2,2), padding='same')(f)
    f = BatchNormalization()(f)
    f = LeakyReLU()(f)


    f = Conv2D(32, (3,3), strides=(2,2), padding='same')(f)
    f = BatchNormalization()(f)
    f = LeakyReLU()(f)

    return Model(input_layer, f)
    
def create_discriminator(feature_extractor,Nt,Nd):
    input_layer = Input(shape=(Nt,Nd,1))

    f = feature_extractor(input_layer)

    d = GlobalAveragePooling2D()(f)
    d = Dense(1, activation='sigmoid')(d)
    
    return Model(input_layer, d)
    
class AdvLoss(Layer):
    def __init__(self, feature_extractor, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)
        self.feature_extractor = feature_extractor

    def call(self, x, mask=None):
        ori_feature = self.feature_extractor(x[0])
        gan_feature = self.feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class CntLoss(Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
def loss(yt, yp):
    return yp

def generator_trainer(generator,feature_extractor,Nt,Nd):
    x = Input(shape=(Nt,Nd,1))
    x_rec = generator(x) 

    adv_loss = AdvLoss(feature_extractor,name='adv_loss')([x, x_rec])
    cnt_loss = CntLoss(name='cnt_loss')([x, x_rec])

    return Model(x, [adv_loss, cnt_loss])
    
def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y]
