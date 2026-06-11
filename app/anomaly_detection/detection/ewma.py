import pandas as pd


def detect_ewma(values, span=5, threshold=2):
    series = pd.Series(values)

    ewma = series.ewm(span=span).mean()

    deviation = abs(series.iloc[-1] - ewma.iloc[-1])

    return deviation > threshold, deviation