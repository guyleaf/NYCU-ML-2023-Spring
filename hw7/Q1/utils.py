import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from scipy.spatial.distance import cdist

KERNEL_TYPES = Literal["none", "rbf", "linear", "poly"]


def load_data(
    files: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], tuple[int, int]]:
    CROPED_IMAGE_SIZE = (40, 40)
    files.sort()

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

        # resize to 40 x 40
        image = image.resize(CROPED_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

        images.append(np.array(image, dtype=np.float64).flatten() / 255.0)
        image.close()

    images = np.stack(images, axis=0)
    return images, labels, CROPED_IMAGE_SIZE


def kernel(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    kernel_type: KERNEL_TYPES,
    center: bool = False,
) -> npt.NDArray[np.float64]:
    gamma = 1 / x.shape[1]
    if kernel_type == "rbf":
        kernel = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    else:
        kernel = x @ y.T
        if kernel_type == "poly":
            kernel = (gamma * kernel) ** 3

    if center:
        # center the matrix
        # K_C = K - 1_N * K - K * 1_N + 1_N * K * 1_N
        all_inv_n_matrix = np.full_like(kernel, 1 / kernel.shape[0])
        kernel = (
            kernel
            - all_inv_n_matrix @ kernel
            - kernel @ all_inv_n_matrix
            + all_inv_n_matrix @ kernel @ all_inv_n_matrix
        )

    return kernel


def plot_faces(file_name: str, samples: npt.NDArray[np.float64], cols_per_row: int):
    fig = plt.figure()
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(math.ceil(samples.shape[0] / cols_per_row), cols_per_row),
        axes_pad=0,
    )

    for i, ax in enumerate(grid):
        ax: plt.Axes
        ax.imshow(samples[i], cmap="gray")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    fig.savefig(file_name)
    plt.close(fig)
