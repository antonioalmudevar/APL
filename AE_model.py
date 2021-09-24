from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, ReLU

def create_model(Nt,Nd):
    # Entrada
    capa_in = Input(shape=(Nt,Nd,1))
    # Encoder
    x = Conv2D(8, (3,3), strides=(2,2), padding='same')(capa_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(16, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(32, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Decoder
    x = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(8, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #Salida
    capa_out = Conv2D(1, (2,2), activation='linear', padding='same')(x)

    return Model(inputs=capa_in, outputs=capa_out)