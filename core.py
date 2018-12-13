from keras.layers import LSTM, Dense
from keras.metrics import MSE
from keras.models import Sequential


def build_lstm_model(input_dims, out_dims_1, out_dims_2, out_dims_3):
    model_ = Sequential()
    model_.add(LSTM(input_shape=(None, input_dims), units=out_dims_1, return_sequences=True, activation='sigmoid'))
    model_.add(LSTM(units=out_dims_2, return_sequences=False, activation='sigmoid'))
    model_.add(LSTM(units=out_dims_3, return_sequences=False, activation='sigmoid'))
    model_.add(Dense(10, activation='tanh', use_bias=True))
    model_.compile(optimizer='adam', loss=MSE, metrics=['mse'])
    model_.summary()
    return model_


def pre_process():
    pass


build_lstm_model(10, 15, 20)
