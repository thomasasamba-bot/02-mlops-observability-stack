from sklearn.ensemble import IsolationForest
import numpy as np


def detect_isolation_forest(values):
    model = IsolationForest(contamination=0.1)

    data = np.array(values).reshape(-1, 1)

    model.fit(data)

    prediction = model.predict(data)

    score = model.decision_function(data)[-1]

    return prediction[-1] == -1, abs(score)