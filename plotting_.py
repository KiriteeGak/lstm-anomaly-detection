from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


class Plotting(object):
    @staticmethod
    def peek_error_distribution(anomaly_scores,
                                save=False,
                                no_show=False,
                                fig_name="figure"):
        # Check whether transformed values follows an gaussian dist.
        digitised = np.digitize(anomaly_scores,
                                np.arange(anomaly_scores.min(), anomaly_scores.max(), 0.5)).reshape(4 * 1335)
        counts = list(Counter(list(digitised)).values())
        plt.bar(np.arange(1, len(counts) + 1), counts, width=1)
        if not no_show:
            plt.show()
        if save:
            plt.savefig("plots/{}.png".format(fig_name))

    @staticmethod
    def anomaly_bars(actual,
                     prediction,
                     anomaly_scores,
                     save=False,
                     no_show=False,
                     fig_name="figure"):

        for i, (a, p, as_) in enumerate(zip(actual, prediction, anomaly_scores)):
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot(a, label='actual')
            axarr[0].plot(p, label='prediction')
            axarr[0].legend()
            axarr[1].bar(x=range(0, len(a)), height=as_, label='anomaly score')
            axarr[1].legend()
            plt.legend(loc='upper right')

            if not no_show:
                plt.show()
            if save:
                plt.savefig("plots/{}_{}.png".format(fig_name, i))
