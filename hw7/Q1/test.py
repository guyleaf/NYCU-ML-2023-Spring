import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import cdist
from tap import Tap

from utils import KERNEL_TYPES, kernel, load_data

TRAIN_FOLDER = "Training"
TEST_FOLDER = "Testing"


class Arguments(Tap):
    root_dir: str
    out_dir: str = "output"

    eigen_path: str = "eigen.npy"
    fisher_path: str = "fisher.npy"

    kernel: KERNEL_TYPES = "none"

    num_eigenfaces: int = 25
    """
    Use num_eigenfaces (first num_components) to project data onto num_eigenfaces dimensional space
    """

    num_reconstructed_faces: int = 10
    """
    With num_eigenfaces to reconstruct num_reconstructed_faces faces
    """

    def process_args(self) -> None:
        if Arguments.eigen_path == self.eigen_path:
            self.eigen_path = os.path.join(self.out_dir, self.eigen_path)
        if Arguments.fisher_path == self.fisher_path:
            self.fisher_path = os.path.join(self.out_dir, self.fisher_path)

    def configure(self) -> None:
        self.add_argument("root_dir")


def plot_faces(file_name: str, samples: npt.NDArray[np.float64], cols_per_row: int):
    fig = plt.figure()
    # visualize eigenfaces
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(math.ceil(samples.shape[0] / cols_per_row), cols_per_row),
        axes_pad=0,
    )

    for i, ax in enumerate(grid):
        ax.imshow(samples[i], cmap="gray")

    fig.savefig(file_name)
    plt.close(fig)


def visualize_faces(
    out_dir: str,
    samples: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    width: int,
    file_name: str = "eigenfaces.jpg",
    num_eigenfaces: int = 25,
    num_reconstructed_faces: int = 10,
    cols_per_row: int = 5,
):
    # D: features
    # E: num_components
    # samples: N by D matrix
    # weights: N by E matrix
    os.makedirs(out_dir, exist_ok=True)

    weights = weights[:, :num_eigenfaces]
    # visualize eigenfaces
    plot_faces(
        os.path.join(out_dir, file_name),
        weights.T.reshape(num_eigenfaces, width, -1),
        cols_per_row,
    )

    # randomly sample reconstructed_faces
    indices = np.random.choice(samples.shape[0], num_reconstructed_faces, replace=False)
    samples = samples[indices]

    # reconstruct faces
    faces = samples @ weights @ weights.T

    # visualize
    plot_faces(
        os.path.join(out_dir, f"reconstructed_{file_name}"),
        faces.reshape(num_reconstructed_faces, width, -1),
        cols_per_row,
    )

    error = cdist(faces, samples, metric="euclidean").diagonal().mean()
    print("Reconstruction Average Error: ", error)


def visualize_kernel_faces(
    out_dir: str,
    kernel: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    num_eigenfaces: int = 25,
    num_reconstructed_faces: int = 10,
):
    # K: target data
    # E: num_components
    # kernel: K by N matrix
    # weights: N by E matrix
    os.makedirs(out_dir, exist_ok=True)

    weights = weights[:, :num_eigenfaces]

    # randomly sample reconstructed_faces
    indices = np.random.choice(kernel.shape[0], num_reconstructed_faces, replace=False)
    kernel = kernel[indices]

    # reconstruct kernel
    reconstructed_kernel = kernel @ weights @ weights.T

    error = cdist(reconstructed_kernel, kernel, metric="euclidean").diagonal().mean()
    print("Reconstruction Average Error (Kernel): ", error)


def preprocess(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], kernel_type: KERNEL_TYPES
):
    if kernel_type != "none":
        # N by N matrix
        x = kernel(x, y, kernel_type)

        # center the matrix
        # K_C = K - 1_N * K - K * 1_N + 1_N * K * 1_N
        all_inv_n_matrix = np.full_like(x, 1 / x.shape[0])
        x = (
            x
            - all_inv_n_matrix @ x
            - x @ all_inv_n_matrix
            + all_inv_n_matrix @ x @ all_inv_n_matrix
        )

    return x


if __name__ == "__main__":
    args = Arguments().parse_args()

    pattern = os.path.join(args.root_dir, TRAIN_FOLDER, "*.pgm")
    files = glob.glob(pattern)

    data, labels = load_data(files)
    width = int(math.sqrt(data.shape[1]))

    data = preprocess(data, data, args.kernel)

    pca_weights = np.load(args.eigen_path)
    fisher_weights = np.load(args.fisher_path)

    if args.kernel != "none":
        visualize_kernel_faces(
            args.out_dir,
            data,
            pca_weights,
            num_eigenfaces=args.num_eigenfaces,
            num_reconstructed_faces=args.num_reconstructed_faces,
        )
    else:
        visualize_faces(
            args.out_dir,
            data,
            pca_weights,
            width,
            num_eigenfaces=args.num_eigenfaces,
            num_reconstructed_faces=args.num_reconstructed_faces,
            file_name="eigenfaces.jpg",
        )

        visualize_faces(
            args.out_dir,
            data,
            fisher_weights,
            width,
            num_eigenfaces=args.num_eigenfaces,
            num_reconstructed_faces=args.num_reconstructed_faces,
            file_name="fisherfaces.jpg",
        )
