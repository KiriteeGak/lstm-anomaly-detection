import warnings
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, TimeDistributed
from keras.metrics import MSE, MAE
from keras.models import Sequential

from seq2seq.models import SimpleSeq2Seq, Seq2Seq

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
    def __init__(self,
                 series_size,
                 feature_size,
                 hidden_dimensions,
                 depth=1,
                 epochs=10,
                 modified_output_size=None,
                 dropout=0.3):
        self.series_size = series_size
        self.feature_size = feature_size
        self.hidden_dimensions = hidden_dimensions
        self.depth = depth
        self.epochs = epochs
        self.modified_output_size = modified_output_size
        self.dropout = dropout

    def build_model(self, **kwargs):
        return Seq2Seq(output_dim=self.feature_size if not kwargs else kwargs['output_size'],
                       input_dim=self.feature_size,
                       input_length=None,
                       output_length=self.series_size,
                       hidden_dim=self.hidden_dimensions,
                       depth=self.depth,
                       dropout=self.dropout)

    def train_model(self, x):
        _model = self.build_model()
        _model.compile(optimizer='adam', loss='mse', metrics=['mse'])

        x_train, x_validate, x_test = SplitData(x=x, split_ratios=(0.7, 0.2)).split_data()

        es = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           restore_best_weights=True)
        _model.fit(x=x_train,
                   y=x_train,
                   epochs=self.epochs,
                   batch_size=4,
                   shuffle=True,
                   callbacks=[es],
                   validation_data=(x_validate, x_validate))

        model_output_decoder_updated = self.build_model(output_size=self.modified_output_size)
        model_output_decoder_updated.set_weights(weights=_model.get_weights())

        return model_output_decoder_updated


class ModelMultiVariateGaussian(object):
    def __init__(self, prediction, actual):
        self.prediction = prediction
        self.actual = actual

    def estimate_errors(self):
        return np.array(self.prediction)-np.array(self.actual)

    def export_mean_covariance(self):
        errors = self.estimate_errors()
        return errors, np.mean(errors), np.cov(errors)

    def return_anomaly_scores(self):
        errors, mean_, covariance = self.export_mean_covariance()
        anomaly_score = np.dot(np.dot((errors-mean_).T, covariance), (errors-mean_))
        return anomaly_score


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    encdec_obj = LSTMEncoderDecoderAnomalyDetection(dropout=0.2,
                                                    series_size=1,
                                                    feature_size=100,
                                                    hidden_dimensions=20,
                                                    depth=1,
                                                    epochs=200,
                                                    modified_output_size=100)
    arr = np.random.rand(100, 1, 100)
    model = encdec_obj.train_model(arr)

    arr_test = arr[-20:].reshape(20, 100)
    pred = model.predict(x=arr_test.reshape(20, 1, 100)).reshape(20, 100)

    # plt.plot(arr[-1].reshape(100, 1))
    # plt.plot(pred)
    # plt.show()

    cls_fit = ModelMultiVariateGaussian(pred, arr_test).return_anomaly_scores()
    print(cls_fit.shape)
