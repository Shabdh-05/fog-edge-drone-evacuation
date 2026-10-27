import numpy as np
from sklearn.cluster import KMeans

class SurvivorClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def cluster(self, survivor_coordinates):
        """
        survivor_coordinates = list of (x, y) tuples
        Example: [(100,200), (120,210), (500,600)]
        """

        data = np.array(survivor_coordinates)

        self.model.fit(data)
        labels = self.model.labels_
        centers = self.model.cluster_centers_

        return labels, centers