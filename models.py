import warnings
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, TimeDistributed
from keras.metrics import MSE
from keras.models import Sequential

from seq2seq.models import Seq2Seq
from scipy.signal import medfilt

from build_data import SplitData, import_sample_data, get_btc_usd_price, load_crypto_pairings_data
from plotting import Plotting


class SimpleLSTMAnomalyDetection(SplitData):
    @staticmethod
    def build_model(feature_size,
                    predict_timestamps,
                    *args,
                    hidden_size=10):
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
                 dropout=0.3,
                 plots=True):
        self.series_size = series_size
        self.feature_size = feature_size
        self.hidden_dimensions = hidden_dimensions
        self.depth = depth
        self.epochs = epochs
        self.modified_output_size = modified_output_size
        self.dropout = dropout
        self.ModelingGaussian = ModelMultiVariateGaussian
        self.trained_model = None
        self._validation_data = None
        self._test_data = None
        self.plotting = plots

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
        x_train, x_validate, x_test = SplitData(x=x, split_ratios=(0.6, 0.3)).split_data()
        self._test_data, self._validation_data = x_test, x_validate
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
        self.trained_model = model_output_decoder_updated
        return model_output_decoder_updated

    def model_errors(self):
        self.ModelingGaussian = ModelMultiVariateGaussian(reconstruction_model=self.trained_model,
                                                          plotting=self.plotting)
        _validate_pred = model.predict(x=self._validation_data).reshape(self._validation_data.shape[0], length)
        self.ModelingGaussian.fit(_validate_pred,
                                  self._validation_data.reshape(self._validation_data.shape[0], length))

    def predict_anomalies(self, test_data=None):
        if not test_data:
            return self.ModelingGaussian.predict(self._test_data)
        return self.ModelingGaussian.predict(test_data)


class ModelMultiVariateGaussian(Plotting):
    def __init__(self, reconstruction_model=None, plotting=True):
        self.covariance = None
        self.mean_values = None
        self.validation_anomaly_scores = None
        self.reconstruction_model = reconstruction_model
        self.smoothed_values = None
        self.plots_ = True
        super().__init__()

    @staticmethod
    def estimate_errors(prediction, actual):
        return np.array(prediction)-np.array(actual)

    def export_mean_covariance(self, predictions, actual, no_values, features):
        errors = self.estimate_errors(predictions, actual).flatten().reshape(1, no_values * features)
        return errors, np.mean(errors).reshape(1, 1), errors.std().reshape(1, 1)

    @staticmethod
    def zipped_isinstance(obj, instances):
        for obj_, instances_ in zip(obj, instances):
            if not isinstance(obj_, instances_):
                return False
        return True

    def fit(self, predict=None, actual=None, mahalanobis=False, errors=None, mean_=None, cov_=None):
        if isinstance(actual, np.ndarray) and not isinstance(predict, np.ndarray)\
                and not predict and self.__class__.__name__ == 'MedianSmoothing':
            predict = self.smoothed_values

        if not self.zipped_isinstance((predict, actual), (np.ndarray, np.ndarray)) and \
                not self.zipped_isinstance((errors, mean_, cov_), (np.ndarray, np.ndarray, np.ndarray)):
            raise ValueError("Unable to find actual or predict values for fitting the gaussian")

        if not self.zipped_isinstance((errors, mean_, cov_), (np.ndarray, np.ndarray, np.ndarray)):
            features, no_values = actual.shape[1], actual.shape[0]

            # This cov_ terms is really standard deviation.
            # TODO: Need to rename, but lot to refactor

            errors, mean_, cov_ = self.export_mean_covariance(predict, actual, no_values, features)
        else:
            features, no_values = errors.shape[0], errors.shape[-1]

        if mahalanobis:
            # Check whether the implementation is right. Blocked till then.
            # Helpful: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Non-degenerate_case

            anomaly_score = np.apply_along_axis(lambda err: np.dot(np.dot((err - mean_).T, np.linalg.inv(cov_)),
                                                                   (err - mean_)),
                                                axis=0,
                                                arr=errors)
        else:
            anomaly_score = np.abs((errors-mean_)/cov_)

        if self.zipped_isinstance((predict, actual), (np.ndarray, np.ndarray)):
            self.validation_anomaly_scores, self.mean_values, self.covariance =\
                np.log(anomaly_score.reshape(1, no_values * features)), mean_, cov_
        else:
            return np.log(anomaly_score.reshape(features, 1, no_values))

    def predict(self, actual):
        if not self.reconstruction_model:
            raise ValueError("Unable to find the model to reconstruct time series")
        pred_ = self.reconstruction_model.predict(actual)
        errors = self.estimate_errors(pred_, actual)
        anomaly_scores = self.fit(errors=errors, mean_=self.mean_values, cov_=self.covariance)

        if not self.plots_:
            return anomaly_scores
        else:
            self.anomaly_bars(actual=actual,
                              prediction=pred_,
                              anomaly_scores=anomaly_scores,
                              save=True,
                              no_show=True,
                              fig_name="testing_delete/figure")


class MedianSmoothing(ModelMultiVariateGaussian, Plotting):
    def __init__(self, data_, kernel_=59):
        super().__init__(reconstruction_model=None)
        self.data_ = data_
        self.kernel_ = kernel_
        self.smoothed_values = self._return_smoothed_curve()

    def _return_smoothed_curve(self, data_addn=None):
        if not isinstance(data_addn, np.ndarray) and not data_addn:
            data_addn = self.data_
        return np.array([medfilt(arr, kernel_size=self.kernel_) for arr in data_addn])

    def predict(self, actual, plots=True):
        smoothed = self._return_smoothed_curve(data_addn=actual)
        errors = self.estimate_errors(smoothed, actual)
        if plots:
            self.anomaly_bars(actual=actual,
                              prediction=smoothed,
                              anomaly_scores=self.fit(errors=errors, mean_=self.mean_values, cov_=self.covariance),
                              save=True,
                              no_show=True)
            return self.fit(errors=errors, mean_=self.mean_values, cov_=self.covariance)


if __name__ == '__main__':
    # data, dps, length = import_sample_data()
    data, dps, length = load_crypto_pairings_data()

    # ==================================================================================
    # ======================== For LSTM based recognition ==============================
    # ==================================================================================

    data = data.reshape(dps, 1, length)
    encdec_obj = LSTMEncoderDecoderAnomalyDetection(dropout=0.2,
                                                    series_size=1,
                                                    feature_size=length,
                                                    hidden_dimensions=10,
                                                    depth=1,
                                                    epochs=200,
                                                    modified_output_size=length,
                                                    plots=True)

    model = encdec_obj.train_model(data)
    encdec_obj.model_errors()
    encdec_obj.predict_anomalies()

    # ==================================================================================
    # ======================== For Median smoothing thingy ==============================
    # ==================================================================================

    x_train, x_validate = SplitData(x=data, split_ratios=(0.7,)).split_data()
    shape_x_validate = x_validate.shape[0]

    model_ = MedianSmoothing(data_=x_train, kernel_=29)
    model_.fit(predict=None, actual=x_train)

    pred = model_.predict(actual=x_validate, plots=True)
