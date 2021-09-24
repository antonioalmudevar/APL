from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv2D, Conv2DTranspose

def create_model(Ng,Nt,Nd):
    # Entrada
    capa_in = Input(shape=(Ng,Nt,Nd,1))
    # Encoder
    x = TimeDistributed(Conv2D(8, (3,3), strides=(2,2), activation='relu', padding='same'))(capa_in)
    x = BatchNormalization()(x)

    x = TimeDistributed(Conv2D(16, (3,3), strides=(2,2), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)

    #
    x = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)

    # Decoder
    x = TimeDistributed(Conv2DTranspose(16, (3,3), strides=(2,2), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(Conv2DTranspose(8, (3,3), strides=(2,2), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)

    #Salida
    capa_out = Conv2D(1, (3,3), activation='linear', padding='same')(x)

    return Model(inputs=capa_in, outputs=capa_out)