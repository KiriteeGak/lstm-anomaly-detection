import numpy as np


class SplitData(object):
    def __init__(self, x, split_ratios=(0.6, 0.9)):
        self.x = np.array(x)
        self.sections = (int(len(x) * split_ratios[0]), int(np.ceil(len(x) * split_ratios[1])))

    def split_data(self):
        return np.split(self.x, indices_or_sections=self.sections)
