import glob
import os

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from tap import Tap

from utils import KERNEL_TYPES, kernel, load_data, plot_faces

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

    k_neighbors: int = 5
    """
    Number of K-Neighbors (KNN)
    """

    def process_args(self) -> None:
        if Arguments.eigen_path == self.eigen_path:
            if self.kernel != "none":
                self.eigen_path = os.path.join(
                    self.out_dir, f"kernel_{self.eigen_path}"
                )
            else:
                self.eigen_path = os.path.join(self.out_dir, self.eigen_path)

        if Arguments.fisher_path == self.fisher_path:
            if self.kernel != "none":
                self.fisher_path = os.path.join(
                    self.out_dir, f"kernel_{self.fisher_path}"
                )
            else:
                self.fisher_path = os.path.join(self.out_dir, self.fisher_path)

    def configure(self) -> None:
        self.add_argument("root_dir")


def visualize_faces(
    out_dir: str,
    samples: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    image_size: tuple[int, int],
    file_name: str = "eigenfaces.jpg",
    num_eigenfaces: int = 25,
    num_reconstructed_faces: int = 10,
    cols_per_row: int = 5,
):
    # D: features
    # E: num_components
    # samples: N by D matrix
    # weights: N by E matrix
    width, height = image_size
    os.makedirs(out_dir, exist_ok=True)

    weights = weights[:, :num_eigenfaces]
    # visualize eigenfaces
    plot_faces(
        os.path.join(out_dir, file_name),
        weights.T.reshape(num_eigenfaces, width, height),
        cols_per_row,
    )

    # randomly sample reconstructed_faces
    indices = np.random.choice(samples.shape[0], num_reconstructed_faces, replace=False)
    samples = samples[indices]

    # visualize original faces
    plot_faces(
        os.path.join(out_dir, f"original_reconstructed_{file_name}"),
        samples.reshape(num_reconstructed_faces, width, height),
        cols_per_row,
    )

    # reconstruct faces
    faces = samples @ weights @ weights.T

    # visualize reconstructed faces
    plot_faces(
        os.path.join(out_dir, f"reconstructed_{file_name}"),
        faces.reshape(num_reconstructed_faces, width, height),
        cols_per_row,
    )

    error = cdist(faces, samples, metric="euclidean").diagonal().mean()
    print("Reconstruction Average Error: ", error)


def classify(
    train_data: npt.NDArray[np.float64],
    train_labels: npt.NDArray[np.int32],
    test_data: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    k_neighbors: int = 5,
) -> npt.NDArray[np.int32]:
    # project onto low dimensional space
    train_data = train_data @ weights
    test_data = test_data @ weights

    # find labels of k neighbors
    distance = cdist(test_data, train_data, metric="euclidean")
    k_indices = np.argsort(distance, axis=-1)[:, :k_neighbors]
    labels = train_labels[k_indices.flatten()].reshape(-1, k_neighbors)

    # find mode of labels
    predictions = []
    for row in labels:
        classes, counts = np.unique(row, return_counts=True)
        predictions.append(classes[np.argmax(counts)])

    return np.stack(predictions, axis=0)


def evaluate(
    train_data: npt.NDArray[np.float64],
    train_labels: npt.NDArray[np.int32],
    test_data: npt.NDArray[np.float64],
    test_labels: npt.NDArray[np.int32],
    pca_weights: npt.NDArray[np.float64],
    fisher_weights: npt.NDArray[np.float64],
    k_neighbors: int,
):
    pca_predictions = classify(
        train_data, train_labels, test_data, pca_weights, k_neighbors
    )
    fisher_predictions = classify(
        train_data, train_labels, test_data, fisher_weights, k_neighbors
    )

    print("K-NN Error rate (PCA):", np.mean(pca_predictions != test_labels))
    print("K-NN Error rate (Fisher):", np.mean(fisher_predictions != test_labels))


def evaluate_kernel(
    train_data: npt.NDArray[np.float64],
    train_labels: npt.NDArray[np.int32],
    test_data: npt.NDArray[np.float64],
    test_labels: npt.NDArray[np.int32],
    pca_weights: npt.NDArray[np.float64],
    fisher_weights: npt.NDArray[np.float64],
    k_neighbors: int,
    kernel_type: KERNEL_TYPES,
):
    # calculate kernels
    train_kernel = kernel(train_data, train_data, kernel_type)
    test_kernel = kernel(test_data, train_data, kernel_type)

    pca_predictions = classify(
        train_kernel,
        train_labels,
        test_kernel,
        pca_weights,
        k_neighbors,
    )
    fisher_predictions = classify(
        train_kernel, train_labels, test_kernel, fisher_weights, k_neighbors
    )

    print("K-NN Error rate (PCA):", np.mean(pca_predictions != test_labels))
    print("K-NN Error rate (Fisher):", np.mean(fisher_predictions != test_labels))


def load_dataset(root_dir: str, subfolder: str):
    pattern = os.path.join(root_dir, subfolder, "*.pgm")
    files = glob.glob(pattern)

    data, labels, image_size = load_data(files)
    return data, labels, image_size


if __name__ == "__main__":
    args = Arguments().parse_args()

    # load data
    train_data, train_labels, image_size = load_dataset(args.root_dir, TRAIN_FOLDER)
    test_data, test_labels, _ = load_dataset(args.root_dir, TEST_FOLDER)

    # load weights
    pca_weights = np.load(args.eigen_path)
    fisher_weights = np.load(args.fisher_path)

    if args.kernel == "none":
        visualize_faces(
            args.out_dir,
            train_data,
            pca_weights,
            image_size,
            num_eigenfaces=args.num_eigenfaces,
            num_reconstructed_faces=args.num_reconstructed_faces,
            file_name="eigenfaces.jpg",
        )
        visualize_faces(
            args.out_dir,
            train_data,
            fisher_weights,
            image_size,
            num_eigenfaces=args.num_eigenfaces,
            num_reconstructed_faces=args.num_reconstructed_faces,
            file_name="fisherfaces.jpg",
        )
        evaluate(
            train_data,
            train_labels,
            test_data,
            test_labels,
            pca_weights,
            fisher_weights,
            args.k_neighbors,
        )
    else:
        evaluate_kernel(
            train_data,
            train_labels,
            test_data,
            test_labels,
            pca_weights,
            fisher_weights,
            args.k_neighbors,
            args.kernel,
        )
