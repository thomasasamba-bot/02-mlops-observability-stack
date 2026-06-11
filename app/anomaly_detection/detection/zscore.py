import numpy as np


def detect_zscore(values, threshold=3):
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return False, 0

    zscore = abs((values[-1] - mean) / std)

    return zscore > threshold, zscore