from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.spatial.distance import cdist

KERNEL_TYPES = Literal["none", "rbf", "linear", "poly"]


def load_data(
    files: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    text = " ".join(files)
    number_of_images_per_subjects = text.count("subject01")
    number_of_subjects = len(files) // number_of_images_per_subjects

    labels = np.repeat(list(range(number_of_subjects)), number_of_images_per_subjects)

    images = []
    for file in files:
        image = Image.open(file)
        width, height = image.size

        # center crop
        short_side = min(width, height)
        left = (width - short_side) / 2
        top = (height - short_side) / 2
        right = width - left
        bottom = height - top
        image = image.crop((round(left), round(top), round(right), round(bottom)))

        images.append(np.array(image, dtype=np.float64).flatten() / 255.0)
        image.close()

    images = np.stack(images, axis=0)
    return images, labels


def kernel(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES
) -> npt.NDArray[np.float64]:
    gamma = 1 / x.shape[1]
    if kernel_type == "rbf":
        return np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    else:
        kernel = x @ y.T
        if kernel_type == "poly":
            return (gamma * kernel) ** 3
        return kernel
