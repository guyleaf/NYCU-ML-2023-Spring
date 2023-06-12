import glob
import os
import timeit

import numpy as np
import numpy.typing as npt
from scipy.linalg import inv
from scipy.sparse.linalg import eigsh
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
    eigenvalues, eigenvectors = eigsh(x, k=num_components)

    print(
        "Solving eigenvectors in",
        timeit.default_timer() - start_time,
        "seconds",
    )

    eigenvectors = eigenvectors[:, ::-1]
    return eigenvectors


def solve_by_fisher(
    x: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int32],
    kernel_type: KERNEL_TYPES,
    num_components: int,
) -> npt.NDArray[np.float64]:
    # x: N by D matrix (normal)
    # x: N by N matrix (kernel)
    # labels: N array

    if num_components > x.shape[1]:
        text = "the number of features"
        if kernel_type != "none":
            text = "the number of samples"
        raise ValueError(f"The num_components should not be larger than {text}.")

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
            # covariance_within_class += distance_within_class.T @ distance_within_class
            for distance in distance_within_class[:, None]:
                covariance_within_class += distance.T @ distance
        else:
            # within-class covariance: kernel
            group_identity = np.identity(counts[i])
            covariance_within_class += (
                group_x.T
                @ (group_identity - np.full_like(group_identity, 1 / counts[i]))
                @ group_x
            )

    # S_w^-1 * S_b
    inversed_covariance_within_class = inv(covariance_within_class, overwrite_a=True)
    start_time = timeit.default_timer()
    eigenvalues, eigenvectors = eigsh(
        inversed_covariance_within_class @ covariance_between_class, k=num_components
    )

    print(
        "Solving eigenvectors in",
        timeit.default_timer() - start_time,
        "seconds",
    )

    eigenvectors = eigenvectors[:, ::-1]
    return eigenvectors


def preprocess(x: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES):
    if kernel_type != "none":
        # N by N matrix
        x = kernel(x, x, kernel_type)

        # center the matrix
        # K_C = K - 1_N * K - K * 1_N + 1_N * K * 1_N
        all_inv_n_matrix = np.full_like(x, 1 / x.shape[0])
        x = (
            x
            - all_inv_n_matrix @ x
            - x @ all_inv_n_matrix
            + all_inv_n_matrix @ x @ all_inv_n_matrix
        )
    else:
        # D by D matrix
        x = np.cov(x, rowvar=False, bias=True)

    return x


if __name__ == "__main__":
    args = Arguments().parse_args()

    pattern = os.path.join(args.root_dir, TRAIN_FOLDER, "*.pgm")
    files = glob.glob(pattern)

    data, labels = load_data(files)
    x = preprocess(data, args.kernel)

    if args.kernel != "none":
        pca_weights = solve_by_pca(x, args.kernel, args.num_components)
        fisher_weights = solve_by_fisher(
            x, labels, args.kernel, args.num_components
        )
        np.save(os.path.join(args.out_dir, "eigen_kernel.npy"), pca_weights)
        np.save(os.path.join(args.out_dir, "fisher_kernel.npy"), fisher_weights)
    else:
        pca_weights = solve_by_pca(x, args.kernel, args.num_components)
        fisher_weights = solve_by_fisher(data, labels, args.kernel, args.num_components)
        np.save(os.path.join(args.out_dir, "eigen.npy"), pca_weights)
        np.save(os.path.join(args.out_dir, "fisher.npy"), fisher_weights)
