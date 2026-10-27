import numpy as np
from sklearn.linear_model import LogisticRegression

class RiskScorer:
    def __init__(self):
        # Logistic Regression model
        self.model = LogisticRegression()
        self._train_dummy_model()

    def _train_dummy_model(self):
        """
        Train on simulated data:
        Features:
        [cluster_size, hazard_severity, distance_to_hazard]
        """

        X = np.array([
            [3, 8, 50],
            [5, 9, 40],
            [2, 7, 60],
            [8, 3, 200],
            [6, 2, 180],
            [4, 1, 220],
        ])

        # 1 = High Risk, 0 = Low Risk
        y = np.array([1, 1, 1, 0, 0, 0])

        self.model.fit(X, y)

    def predict_risk(self, cluster_size, hazard_severity, distance):
        features = np.array([[cluster_size, hazard_severity, distance]])
        prediction = self.model.predict(features)

        return "High Risk" if prediction[0] == 1 else "Low Risk"