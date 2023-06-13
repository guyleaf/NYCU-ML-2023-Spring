import glob
import os
import timeit

import numpy as np
import numpy.typing as npt
from tap import Tap

from utils import KERNEL_TYPES, kernel, load_data

TRAIN_FOLDER = "Training"
TEST_FOLDER = "Testing"


class Arguments(Tap):
    root_dir: str
    out_dir: str = "output"

    kernel: KERNEL_TYPES = "none"

    num_components: int = 25

    def configure(self) -> None:
        self.add_argument("root_dir")


def solve_by_pca(
    x: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES, num_components: int
) -> npt.NDArray[np.float64]:
    # x: D by D matrix (normal)
    # x: N by N matrix (kernel)

    if num_components > x.shape[0]:
        text = "the number of features"
        if kernel_type != "none":
            text = "the number of samples"
        raise ValueError(f"The num_components should not be larger than {text}.")

    start_time = timeit.default_timer()

    eigenvalues, eigenvectors = np.linalg.eigh(x)

    print(
        "Finished solving eigenvectors in",
        timeit.default_timer() - start_time,
        "seconds",
    )

    indices = np.argsort(eigenvalues)[: -num_components - 1 : -1]
    eigenvectors = eigenvectors[:, indices]
    return eigenvectors


def calculate_between_and_with_class_covariance(
    x: npt.NDArray[np.float64], labels: npt.NDArray[np.int32], kernel_type: KERNEL_TYPES
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # calculate global mean: normal + kernel
    global_mean: npt.NDArray[np.float64] = np.mean(x, axis=0, keepdims=True)

    # calculate with-class & between-class covariance
    groups, counts = np.unique(labels, return_counts=True)
    covariance_between_class = np.zeros((x.shape[1],) * 2, dtype=np.float64)
    covariance_within_class = np.zeros_like(covariance_between_class)
    for i in groups:
        group_x = x[labels == i]
        group_mean: npt.NDArray[np.float64] = np.mean(group_x, axis=0, keepdims=True)

        # between-class covariance: normal + kernel
        centered_group_mean = group_mean - global_mean
        covariance_between_class += (
            counts[i] * centered_group_mean.T @ centered_group_mean
        )

        if kernel_type == "none":
            # within-class covariance: normal
            centered_group_x = group_x - group_mean
            covariance_within_class += np.dot(centered_group_x.T, centered_group_x)
        else:
            # within-class covariance: kernel
            group_identity = np.identity(counts[i])
            covariance_within_class += (
                group_x.T
                @ (group_identity - np.full_like(group_identity, 1 / counts[i]))
                @ group_x
            )

    return covariance_between_class, covariance_within_class


def solve_by_fisher(
    x: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int32],
    kernel_type: KERNEL_TYPES,
    num_components: int,
) -> npt.NDArray[np.float64]:
    # x: N by D matrix
    # labels: N array

    if num_components > x.shape[1]:
        text = "the number of features"
        if kernel_type != "none":
            text = "the number of samples"
        raise ValueError(f"The num_components should not be larger than {text}.")

    start_time = timeit.default_timer()
    (
        covariance_between_class,
        covariance_within_class,
    ) = calculate_between_and_with_class_covariance(x, labels, kernel_type)

    print(
        "Finished calculating covariances in",
        timeit.default_timer() - start_time,
        "seconds",
    )

    # S_w^-1 * S_b
    start_time = timeit.default_timer()

    inversed_covariance_within_class = np.linalg.pinv(covariance_within_class)
    eigenvalues, eigenvectors = np.linalg.eigh(
        inversed_covariance_within_class @ covariance_between_class
    )

    print(
        "Finished solving eigenvectors in",
        timeit.default_timer() - start_time,
        "seconds",
    )

    indices = np.argsort(eigenvalues)[: -num_components - 1 : -1]
    eigenvectors = eigenvectors[:, indices]
    return eigenvectors


def preprocess(x: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES):
    centered_x = None
    if kernel_type != "none":
        # N by N matrix
        x = kernel(x, x, kernel_type)
        centered_x = kernel(x, x, kernel_type, center=True)
    else:
        # D by D matrix
        x = np.cov(x, rowvar=False, bias=True)

    return x, centered_x


if __name__ == "__main__":
    args = Arguments().parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pattern = os.path.join(args.root_dir, TRAIN_FOLDER, "*.pgm")
    files = glob.glob(pattern)

    data, labels, _ = load_data(files)
    x, centered_x = preprocess(data, args.kernel)

    if args.kernel != "none":
        pca_weights = solve_by_pca(centered_x, args.kernel, args.num_components)
        fisher_weights = solve_by_fisher(x, labels, args.kernel, args.num_components)
        np.save(os.path.join(args.out_dir, "kernel_eigen.npy"), pca_weights)
        np.save(os.path.join(args.out_dir, "kernel_fisher.npy"), fisher_weights)
    else:
        pca_weights = solve_by_pca(x, args.kernel, args.num_components)
        fisher_weights = solve_by_fisher(data, labels, args.kernel, args.num_components)
        np.save(os.path.join(args.out_dir, "eigen.npy"), pca_weights)
        np.save(os.path.join(args.out_dir, "fisher.npy"), fisher_weights)
