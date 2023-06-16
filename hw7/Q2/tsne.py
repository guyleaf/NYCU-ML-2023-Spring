#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import argparse
import os
from enum import Enum

import numpy as np
import pylab
from PIL import Image

OUT_DIR = "output"


class Mode(Enum):
    symmetric_sne = "s-sne"
    t_sne = "t-sne"

    def __str__(self):
        return self.value


def visualize_similarities(
    P: np.ndarray, Q: np.ndarray, labels: np.ndarray, perplexity: float, mode: Mode
):
    # reorder rows and cols by labels
    # block-wise similarity matrix
    indices = np.argsort(labels)
    P = P[indices][:, indices]
    Q = Q[indices][:, indices]

    min_P = np.min(P)
    scaled_P = (P - min_P) / (np.max(P) - min_P)

    min_Q = np.min(Q)
    scaled_Q = (Q - min_Q) / (np.max(Q) - min_Q)

    fig = pylab.figure()
    ax1, ax2 = fig.subplots(1, 2)

    fig.suptitle(f"Algorithm = {str(mode)}, Perplexity = {perplexity}")

    ax1.set_title("High-dimensional space")
    image = ax1.matshow(scaled_P, cmap="hot")
    ax1.set_xlabel("label")
    ax1.set_ylabel("label")
    fig.colorbar(image)

    ax2.set_title("Low-dimensional space")
    image = ax2.matshow(scaled_Q, cmap="hot")
    ax2.set_xlabel("label")
    ax2.set_ylabel("label")
    fig.colorbar(image)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{str(mode)}_similarities.png"))

    pylab.close(fig)


def draw_results(
    step: int, Y: np.ndarray, labels: np.ndarray, perplexity: float, mode: Mode
):
    fig = pylab.figure()
    ax1 = fig.subplots(1, 1)

    fig.suptitle(f"Step = {step}, Algorithm = {str(mode)}, Perplexity = {perplexity}")

    ax1.scatter(Y[:, 0], Y[:, 1], 20, labels)

    fig.canvas.draw()
    image = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )

    pylab.close(fig)
    return image


def Hbeta(D=np.array([]), beta=1.0):
    """
    Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
    Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
    Runs PCA on the NxD array X in order to reduce its dimensionality to
    no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def sne(
    X=np.array([]),
    no_dims=2,
    initial_dims=50,
    perplexity=30.0,
    mode: Mode = Mode.symmetric_sne,
):
    """
    Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if not np.issubdtype(X.dtype, np.floating):
        print("Error: array X should be a float array.")
        return -1
    if isinstance(no_dims, float):
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    history_images: list[Image.Image] = []

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2.0 * np.dot(Y, Y.T)
        if mode == Mode.symmetric_sne:
            num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if mode == Mode.symmetric_sne:
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            else:
                dY[i, :] = np.sum(
                    np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0
                )

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)) + (gains * 0.8) * (
            (dY > 0.0) == (iY > 0.0)
        )
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            history_images.append(draw_results(iter + 1, Y, labels, perplexity, mode))
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.0

    history_images[0].save(
        os.path.join(OUT_DIR, f"{str(mode)}_history.gif"),
        save_all=True,
        append_images=history_images[1:],
        optimize=False,
        loop=0,
        duration=300,
    )
    visualize_similarities(P, Q, labels, perplexity, mode)

    # Return solution
    return Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algorithm",
        type=Mode,
        choices=list(Mode),
        default=Mode.symmetric_sne,
        help="Choose an algorithm to perform dimensional rediction",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=20.0,
        help="perplexity",
    )
    args = parser.parse_args()

    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("data/mnist2500_X.txt")
    labels = np.loadtxt("data/mnist2500_labels.txt")

    os.makedirs(OUT_DIR, exist_ok=True)

    Y = sne(
        X, no_dims=2, initial_dims=50, perplexity=args.perplexity, mode=args.algorithm
    )
    pylab.title(str(args.algorithm))
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.savefig(os.path.join(OUT_DIR, f"{str(args.algorithm)}_final.png"))
    pylab.close()
