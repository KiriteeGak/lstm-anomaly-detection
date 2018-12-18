import numpy as np


class SplitData(object):
    def __init__(self, x, split_ratios=(0.7, 0.1), axis=0):
        self.x = np.array(x)
        self.axis = axis
        self.sections = tuple([int(x.shape[axis] *
                                   round(sum(split_ratios[:i+1]), 1)) for i, _ in enumerate(split_ratios)])

    def split_data(self):
        return np.split(self.x, indices_or_sections=self.sections, axis=self.axis)

    def create_data(self):
        pass
