import warnings
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, TimeDistributed
from keras.metrics import MSE
from keras.models import Sequential

from seq2seq.models import SimpleSeq2Seq

from build_data import SplitData


class SimpleLSTMAnomalyDetection(SplitData):
    @staticmethod
    def build_model(feature_size, predict_timestamps, *args, hidden_size=10):
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

    def train_model(self):
        pass

    def early_stopping(self):
        pass

    def predict_anomalies(self):
        pass


class LSTMEncoderDecoderAnomalyDetection(object):
    def __init__(self, series_size, feature_size, hidden_dimensions, depth=1, epochs=10, modified_output_size=None):
        self.series_size = series_size
        self.feature_size = feature_size
        self.hidden_dimensions = hidden_dimensions
        self.depth = depth
        self.epochs = epochs
        self.modified_output_size = modified_output_size

    def build_model(self, **kwargs):
        return SimpleSeq2Seq(input_dim=self.feature_size,
                             output_dim=self.feature_size,
                             input_length=None,
                             output_length=self.series_size if not kwargs else kwargs['output_size'],
                             hidden_dim=self.hidden_dimensions,
                             depth=self.depth)

    def train_model(self, x):
        _model = self.build_model()
        _model.compile(optimizer='adam', loss='mse', metrics=['mse'])

        x_train, x_validate = SplitData(x=x).split_data()

        es = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=2,
                           restore_best_weights=True)
        _model.fit(x=x_train,
                   y=x_train,
                   epochs=self.epochs,
                   batch_size=4,
                   shuffle=True,
                   callbacks=[es],
                   validation_data=(x_validate, x_validate))

        model_output_decoder_updated = self.build_model(output_size=100)
        model_output_decoder_updated.set_weights(weights=_model.get_weights())

        return model_output_decoder_updated

    def predict_anomalies(self):
        pass


if __name__ == '__main__':
    encdec_obj = LSTMEncoderDecoderAnomalyDetection(series_size=50,
                                                    feature_size=1,
                                                    hidden_dimensions=10,
                                                    depth=1,
                                                    epochs=5,
                                                    modified_output_size=100)
    model = encdec_obj.train_model(np.random.rand(20, 100, 1))
    print(model.predict(x=np.random.rand(1, 40, 1)).shape)
