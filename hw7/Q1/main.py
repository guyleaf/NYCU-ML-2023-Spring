import glob
import os
from typing import Literal

import numba
import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.spatial.distance import cdist
from tap import Tap

TRAIN_FOLDER = "Training"
TEST_FOLDER = "Testing"

KERNEL_TYPES = Literal["none", "rbf", "linear", "poly"]


class Arguments(Tap):
    root_dir: str
    out_dir: str = "output"

    kernel: KERNEL_TYPES = "none"

    num_eigenvectors: int = 10000  # k eigenvectors == size of projected image
    num_eigenfaces: int = 25
    num_reconstructed_faces: int = 10

    def configure(self) -> None:
        self.add_argument("root_dir")


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

        images.append(np.array(image).flatten())
        image.close()
    return np.stack(images, axis=0), labels


def kernel(
    x: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES
) -> npt.NDArray[np.float64]:
    gamma = 1 / x.shape[1]
    if kernel_type == "rbf":
        return np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    else:
        kernel = x @ x.T
        if kernel_type == "poly":
            return (gamma * kernel) ** 3
        return kernel


@numba.njit()
def solve_by_pca(
    x: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES, num_eigenvectors: int
) -> npt.NDArray[np.float64]:
    # x: D by D matrix (normal)
    # x: N by N matrix (kernel)

    if num_eigenvectors > x.shape[0]:
        text = "the number of features"
        if kernel_type != "none":
            text = "the number of samples"
        raise ValueError(f"The num_eigenvectors should not be larger than {text}.")

    eigenvalues, eigenvectors = np.linalg.eigh(x)

    indices = np.argsort(eigenvalues)[: -(num_eigenvectors + 1) : -1]
    eigenvectors = eigenvectors[:, indices]
    return eigenvectors


@numba.njit(parallel=True)
def solve_by_fisher(
    x: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int32],
    kernel_type: KERNEL_TYPES,
    num_eigenvectors: int,
) -> npt.NDArray[np.float64]:
    # x: N by D matrix (normal)
    # x: N by N matrix (kernel)
    # labels: N array

    if num_eigenvectors > x.shape[1]:
        text = "the number of features"
        if kernel_type != "none":
            text = "the number of samples"
        raise ValueError(f"The num_eigenvectors should not be larger than {text}.")

    # calculate global mean: normal + kernel
    global_mean: npt.NDArray[np.float64] = x.mean(axis=0, keepdims=True)

    # calculate with-class & between-class covariance
    groups, counts = np.unique(labels, return_counts=True)
    covariance_between_class = np.zeros((x.shape[1],) * 2, dtype=np.float64)
    covariance_within_class = np.zeros_like(covariance_between_class)
    for i in groups:
        group_x = x[labels == i]
        group_mean: npt.NDArray[np.float64] = group_x.mean(axis=0, keepdims=True)

        # between-class covariance: normal + kernel
        distance_between_class = group_mean - global_mean
        covariance_between_class += (
            counts[i] * distance_between_class.T @ distance_between_class
        )

        if kernel_type == "none":
            # within-class covariance: normal
            distance_within_class = group_x - group_mean
            covariance_within_class += distance_within_class.T @ distance_within_class
        else:
            # within-class covariance: kernel
            group_identity = np.identity(counts[i])
            covariance_within_class += (
                group_x.T
                @ (group_identity - np.full_like(group_identity, 1 / counts[i]))
                @ group_x
            )

    # S_w^-1 * S_b
    inversed_covariance_within_class = np.linalg.inv(covariance_within_class)
    eigenvalues, eigenvectors = np.linalg.eigh(
        inversed_covariance_within_class @ covariance_between_class
    )

    indices = np.argsort(eigenvalues)[: -(num_eigenvectors + 1) : -1]
    eigenvectors = eigenvectors[:, indices]
    return eigenvectors


def preprocess(x: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES):
    normalized_x = None
    if kernel_type != "none":
        # N by N matrix
        x = kernel(x, kernel_type)

        # center the matrix
        # K_C = K - 1_N * K - K * 1_N + 1_N * K * 1_N
        all_inv_n_matrix = np.full_like(x, 1 / x.shape[0])
        normalized_x = (
            x
            - all_inv_n_matrix @ x
            - x @ all_inv_n_matrix
            + all_inv_n_matrix @ x @ all_inv_n_matrix
        )
    else:
        # D by D matrix
        x = np.cov(x, rowvar=False, bias=True)

    return x, normalized_x


if __name__ == "__main__":
    args = Arguments().parse_args()

    pattern = os.path.join(args.root_dir, TRAIN_FOLDER, "*.pgm")
    files = glob.glob(pattern)

    images, labels = load_data(files)

    data = images.astype(np.float64)
    x, normalized_x = preprocess(data, args.kernel)

    if args.kernel != "none":
        pca_weights = solve_by_pca(normalized_x, args.kernel, args.num_eigenvectors)
        fisher_weights = solve_by_fisher(
            normalized_x, labels, args.kernel, args.num_eigenvectors
        )
    else:
        pca_weights = solve_by_pca(x, args.kernel, args.num_eigenvectors)
        fisher_weights = solve_by_fisher(
            data, labels, args.kernel, args.num_eigenvectors
        )

    print("test")
