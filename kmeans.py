from sklearn.cluster import KMeans
import numpy as np


def predict(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
    kmeans.fit(data)
    return (kmeans.labels_, kmeans.cluster_centers_)
