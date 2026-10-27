import numpy as np
from sklearn.cluster import DBSCAN

class DensityAnalyzer:
    def __init__(self, eps=50, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples

    def compute_density_score(self, positions):
        if len(positions) == 0:
            return 0.0, []

        points = np.array(positions)

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_

        cluster_density = {}
        for label in labels:
            if label == -1:
                continue
            cluster_density[label] = cluster_density.get(label, 0) + 1

        max_density = max(cluster_density.values()) if cluster_density else 0

        density_score = min(max_density / 10.0, 1.0)

        return round(density_score, 2), labels