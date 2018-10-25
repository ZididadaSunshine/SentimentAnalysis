impo
from keras import Input, Model
from keras.initializers import Ones
from keras.layers import Conv2D, LSTM, Dropout, Activation, Reshape, Masking

input = Input(shape=(300,256,1))

conv1 = Conv2D(filters=1, kernel_size=(300, 3), strides=(300,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(input)
dropout = Dropout(0.5)(conv1)
reshape = Reshape((1, 256))(dropout)
masking = Masking(mask_value=0, input_shape=(1, 256))(reshape)
lstm = LSTM(units=(2), activation='tanh', use_bias=True)(masking)
act = Activation('sigmoid')(lstm)

model = Model(inputs=[input], outputs=[act])
model.summary()