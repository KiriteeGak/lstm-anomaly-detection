import warnings
import numpy as np

from keras.layers import LSTM, Dense, TimeDistributed
from keras.metrics import MSE
from keras.models import Sequential


def build_lstm_model(feature_size, predict_timestamps, *args, hidden_size=10):
    model_ = Sequential()

    if args:
        if isinstance(args, (list, tuple)):
            for layer_index, dims in enumerate(args):
                if not layer_index:
                    model_.add(LSTM(input_shape=(None, feature_size),
                                    units=dims,
                                    return_sequences=True,
                                    activation='sigmoid'))
                else:
                    model_.add(LSTM(units=dims,
                                    return_sequences=True,
                                    activation='sigmoid'))
        elif isinstance(args[0], (dict,)):
            for layer_index, layer_attributes in enumerate(args):
                if 'dimensions' not in layer_attributes.keys() or 'activation' not in layer_attributes.keys():
                    raise KeyError("Unable to find attributes need to build an lstm layer."
                                   "Attributes are `activation` and `size`")
                elif not layer_index:
                    model_.add(LSTM(input_shape=(None, feature_size),
                                    units=layer_attributes['dimensions'],
                                    return_sequences=True,
                                    activation=layer_attributes['activation']))
                else:
                    model_.add(LSTM(units=layer_attributes['dimensions'],
                                    return_sequences=True,
                                    activation=layer_attributes['activation']))
    else:
        warnings.warn("\nNot able to build stacking layers. "
                      "Hidden dimensions of lstm are set to 10. "
                      "Attribute can be set by using `hidden_size`. "
                      "Will not be used when using one more stacking lstm layers.")

        model_.add(LSTM(input_shape=(None, feature_size),
                        units=hidden_size,
                        return_sequences=True,
                        activation='sigmoid'))

    model_.add(TimeDistributed(Dense(predict_timestamps)))
    model_.compile(optimizer='adam', loss=MSE, metrics=['mse'])
    model_.summary()
    return model_


def build_data():
    pass


def split_timestamps():
    pass


if __name__ == '__main__':
    model = build_lstm_model(10, 20, 10, 10, 20, hidden_size=20)
    model.fit(x=np.random.rand(1, 20, 10), y=np.random.rand(1, 20, 20), epochs=10, verbose=0)
    y = model.predict(x=np.random.rand(1, 1, 10))
    print(y)
