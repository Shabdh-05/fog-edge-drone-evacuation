from ml_modules.clustering import SurvivorClustering

# Dummy survivor coordinates (simulate YOLO output)
survivors = [
    (100, 200),
    (120, 210),
    (130, 190),
    (800, 900),
    (820, 880),
    (790, 910)
]

clusterer = SurvivorClustering(n_clusters=2)
labels, centers = clusterer.cluster(survivors)

print("Cluster Labels:", labels)
print("Cluster Centers:", centers)