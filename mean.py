from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np


def predict(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_samples = data.shape[0] * data.shape[1]
    band = estimate_bandwidth(data, quantile=0.2, n_samples=n_samples)

    ms_model = MeanShift(bandwidth=band, bin_seeding=True)
    ms_model.fit(data)
    return (ms_model.labels_, ms_model.cluster_centers_)
