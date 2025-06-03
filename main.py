import numpy as np
import matplotlib.pyplot as plt
from mean import predict as prd
from PIL import Image


def load(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    image_array = np.array(image)
    return image_array


def predict(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return prd(data=data)


def draw(data: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    plt.imshow(data, cmap="gray")
    plt.scatter(centers[:, 0], centers[:, 1], s=250, c="blue", marker="X")
    plt.show()


if __name__ == "__main__":
    data = load("test.jpg")
    (labels, centers) = predict(data)
    draw(data, labels, centers)
