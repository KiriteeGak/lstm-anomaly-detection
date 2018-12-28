import numpy as np
import random as rd
import pandas as pd


class SplitData(object):
    def __init__(self, x, split_ratios=(0.6, 0.3), axis=0):
        self.x = x
        rd.shuffle(np.array(self.x))
        self.axis = axis
        self.sections = tuple([int(x.shape[axis] *
                                   round(sum(split_ratios[:i+1]), 1)) for i, _ in enumerate(split_ratios)])

    def split_data(self):
        return np.split(self.x, indices_or_sections=self.sections, axis=self.axis)

    def create_data(self):
        pass


def import_sample_data():
    # Load market data
    data_loaded = pd.read_csv("data/market.csv")
    # data_loaded = data_loaded[data_loaded['e_date'] < '2017-07-17']

    # Assuming they are of opening prices
    data_loaded.fillna(method='bfill', inplace=True)
    data_loaded.dropna(how='any', axis=0, inplace=True)
    data_loaded_slice = data_loaded[data_loaded.columns.difference(['e_date'])].apply(pd.to_numeric)

    data_loaded[data_loaded_slice.columns] = \
        (data_loaded_slice - data_loaded_slice.min()) / (data_loaded_slice.max() - data_loaded_slice.min())

    if False:
        data_loaded[data_loaded_slice.columns] = data_loaded[data_loaded_slice.columns].diff()
        data_loaded.dropna(how='any', axis=0, inplace=True)

    data_ = data_loaded[data_loaded_slice.columns.difference(['e_date', 'BITFINEX_SPOT_BTC_USD'])].values.T
    dps_, length_ = np.shape(data_)
    return data_, dps_, length_


def get_btc_usd_price():
    d = pd.read_csv("data/market.csv")['BITFINEX_SPOT_BTC_USD']
    d.fillna(method='bfill', inplace=True)
    d = d.values
    return (d-d.min())/(d.max()-d.min())