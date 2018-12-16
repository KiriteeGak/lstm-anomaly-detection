import numpy as np


class SplitData(object):
    def __init__(self, x, split_ratios=(0.5,), axis=1):
        self.x = np.array(x)
        self.axis = axis
        self.sections = (int(x.shape[self.axis] * split_ratios[0]),)

    def split_data(self):
        return np.split(self.x, indices_or_sections=self.sections, axis=1)

    def create_data(self):
        pass
