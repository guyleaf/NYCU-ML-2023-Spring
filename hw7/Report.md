# Machine Learning Homework 7

[TOC]

## Environment

* Language: Python

* Version: 3.9.16

## Code with detailed explanations

### Kernel Eigenfaces

#### Part1

* Algorithms

  * PCA

    1. Calculate covariance matrix of data $X$ (D by D matrix)

       ![image-20230614155329162](assets/pca_cov.png)

    2. According to *Rayleigh quotient*,
       we could use eigen decomposition on covariance matrix of data to get the first k largest eigenvectors (principal components) as $W$ orthogonal projection matrix

       ![image-20230614155854177](assets/pca_eigh.png)

    3. Project the data onto low dimensional space

       $Z = XW$

    4. Reconstruct the data (lossless if # of eigenvectors = D)

       $X = ZW^T$

  * LDA

    1. Within-class scatter $S_W$

       $\bold{m}_j = \dfrac{1}{n_j} \sum_{i \in C_j} x_i$

       $S_W = \sum_{j=1}^k \sum_{i \in C_j} (x_i - \bold{m}_j) (x_i - \bold{m}_j)^T$

    2. Between-class scatter $S_B$

       $\bold{m} = \dfrac{1}{n} \sum x$

       $S_B = \sum_{j=1}^k (\bold{m}_j - \bold{m}) (\bold{m}_j - \bold{m})^T$

    3. According to the objective function $J(W) = \dfrac{\det(W^TS_BW)}{\det(W^TS_WW)}$ and *Rayleigh quotient*,
       we could formulate it to

       $S_Bw_l = \lambda_lS_Ww_l$

       $S^{-1}_WS_Bw_l = \lambda_lw_l$

    4. Use eigen decomposition on $S^{-1}_WS_B$ to get the first k largest eigenvectors (principal components) as $W$ orthogonal projection matrix

    5. Project the data onto low dimensional space

       $Z = XW$

    6. Reconstruct the data (lossless if # of eigenvectors = D)

       $X = ZW^T$

* `utils.load_data` function: Load data from PGM files

  > Arguments
  > 	files: list of files, contains PGM files
  >
  > Steps
  >
  > 1. Load the image
  > 2. Center crop image to squared shape
  > 3. Resize the cropped image to 40 by 40 (need less computation resources)
  > 4. Scale it from [0, 255] to [0, 1]

  ![load_data](assets/load_data.png)

* `train.py` main: program entry point

  > In Part1, we only focus on normal PCA and LDA.
  >
  > 
  >
  > Here, the covariance matrix calculation (`np.cov`) corresponds to the first step of PCA.
  >
  > PCA: Calculate covariance matrix of data $X$ (D by D matrix)
  >
  > ![image-20230614155329162](assets/pca_cov.png)

  ![train_normal_main](assets/train_normal_main.png)

* `train.solve_by_pca` function: use PCA algorithm to find $W$ orthogonal projection matrix

  > Arguments
  > 	x: np.ndarray, covariance matrix of data $X$
  > 	kernel_type: ignored
  > 	num_components: keep first k principal components
  >
  > In Part1, we only focus on normal PCA.
  >
  > 
  >
  > Here, it is the second step of PCA.
  >
  > According to *Rayleigh quotient*,
  > we could use eigen decomposition on covariance matrix of data to get the first k largest eigenvectors (principal components) as $W$ orthogonal projection matrix
  >
  > ![image-20230614155854177](assets/pca_eigh.png)

  ![image-20230614172005560](assets/train_pca.png)

* `train.calculate_between_and_with_class_covariance` function: calculate between-class and within-class covariance

  > Arguments
  > 	x: np.ndarray, features of data $X$
  > 	labels: np.ndarray, gt labels of data $X$
  > 	kernel_type: ignored
  > 	num_components: keep first k principal components
  >
  > In Part1, we only focus on normal LDA.
  >
  > 
  >
  > Here, it corresponds to the first and second step of LDA.
  >
  > 1. Within-class scatter $S_W$
  >
  >    $\bold{m}_j = \dfrac{1}{n_j} \sum_{i \in C_j} x_i$
  >
  >    $S_W = \sum_{j=1}^k \sum_{i \in C_j} (x_i - \bold{m}_j) (x_i - \bold{m}_j)^T$
  >
  > 2. Between-class scatter $S_B$
  >
  >    $\bold{m} = \dfrac{1}{n} \sum x$
  >
  >    $S_B = \sum_{j=1}^k (\bold{m}_j - \bold{m}) (\bold{m}_j - \bold{m})^T$

  ![image-20230614173702729](assets/train_lda_cov.png)

* `train.solve_by_fisher` function: use LDA algorithm to find $W$ orthogonal projection matrix

  > Arguments
  > 	x: np.ndarray, covariance matrix of data $X$
  > 	labels: np.ndarray, gt labels of data $X$
  > 	kernel_type: ignored
  > 	num_components: keep first k principal components
  >
  > In Part1, we only focus on normal LDA.
  >
  > 
  >
  > After calculating the between-class and within-class covariance,
  >
  > use eigen decomposition on $S^{-1}_WS_B$ to get the first k largest eigenvectors (principal components) as $W$ orthogonal projection matrix.
  >
  > $S^{-1}_WS_Bw_l = \lambda_lw_l$

  ![image-20230614173242875](assets/train_lda.png)

* `test.py` main: program entry point

  ![image-20230614174147574](assets/test_main.png)

* `utils.plot_faces` function: plot faces on image grid and export image

  > Arguments
  > 	file_name: str, image file name
  > 	samples: np.ndarray, faces
  > 	cols_per_row: int, used for plot image grid

  ![image-20230614175054776](assets/test_plot_faces.png)

* `test.visualize_faces` function: visualize eigenfaces, fisherfaces and reconstructed faces

  > Arguments
  > 	out_dir: str, output folder path
  > 	samples: np.ndarray, shape (N, D), data samples
  > 	weights: PCA or LDA weights, shape (D, E)
  > 	image_size: tuple[int, int], (width, height), data image size
  > 	file_name: str, base output file name
  > 	num_eigenfaces: int, how many eigenfaces need to be visualized
  > 	num_reconstructed_faces: int, how many faces need to be reconstructed
  > 	cols_per_row: int, used for plot image grid
  >
  > 
  >
  > After plot the eigenfaces,
  >
  > we reconstruct faces by the formula $X = XWW^T$.

  ![image-20230614175420953](assets/test_vis_faces.png)

#### Part2

* `test.evaluate` function: evaluate performance by K-NN algorithm on PCA/LDA features

  > Arguments
  > 	train_data: np.ndarray, shape (N, D), train data samples
  > 	train_labels: np.ndarray, shape (N,), train data labels
  > 	test_data: np.ndarray, shape (N, D), test data samples
  > 	test_labels: np.ndarray, shape (N,), test data labels
  > 	pca_weights: PCA weights, shape (D, E)
  > 	fisher_weights: LDA weights, shape (D, E)
  > 	k_neighbors: int, k-NN

  ![image-20230614175538281](assets/test_evaluate.png)

* `test.classify` function: use K-NN algorithm to classify images

  > Arguments
  > 	train_data: np.ndarray, shape (N, D), train data samples
  > 	train_labels: np.ndarray, shape (N,), train data labels
  > 	test_data: np.ndarray, shape (N, D), test data samples
  > 	weights: PCA or LDA weights, shape (D, E)
  > 	k_neighbors: int, k-NN

  ![image-20230614175944578](assets/test_classify.png)

#### Part3

* Algorithms

  * Kernel PCA

    1. Calculate Kernel matrix of data $X$ (N by N matrix), RBF/Linear/Poly

    2. Centered Kernel matrix

       $K^C = K - 1_NK - K1_N + 1_NK1_N$

    3. According to the normal PCA eigenvalue problem in feature space,
       we could use eigen decomposition on centered Kernel matrix to get the first k largest eigenvectors (principal components) as $A$ matrix

       $Ka = \lambda Na$

       $\dfrac{1}{N}Ka = \lambda a$

    4. Project the data onto low dimensional space

       $Z = K(X_{new}, X)A$

  * Kernel LDA

    1. Calculate Kernel matrix of data $X$ (N by N matrix), RBF/Linear/Poly

    2. Within-class scatter $\bold{S}_W$

       $\bold{M}_j = \dfrac{1}{N_j} \sum_{k \in C_j} k(x_k, x_i), i \in N$

       $\bold{S}_W = \sum_{j=1}^k K_j (I - 1_{n_j}) K_j^T$

    3. Between-class scatter $\bold{S}_B$

       $\bold{M} = \dfrac{1}{N} \sum_{k \in N} k(x_k, x_i), i \in N$

       $\bold{S}_B = \sum_{j=1}^k N_j (\bold{M}_j - \bold{M}) (\bold{M}_j - \bold{M})^T$

    4. Use eigen decomposition on $\bold{S}^{-1}_W\bold{S}_B$ to get the first k largest eigenvectors (principal components) as $A$ matrix

    5. Project the data onto low dimensional space

       $Z = K(X_{new}, X)A$

* `utils.kernel` function: calculate Kernel matrix

  > Arguments
  > 	x: np.ndarray, shape (N, D), a set of data
  > 	y: np.ndarray, shape (N, D), another set of data
  > 	kernel_type: Literal["rbf", "linear", "poly"], choose a Kernel to calculate Kernel matrix
  > 	center: center it after calculating Kernel matrix
  >
  > 
  >
  > To center the kernel, we could use this formula.
  >
  > $K^C = K - 1_NK - K1_N + 1_NK1_N$

  ![image-20230614180433506](assets/kernel.png)

* `train.py` main

  > In Part3, we focus on Kernel PCA and LDA.
  >
  > 
  >
  > Kernel PCA assumes the Kernel is centered.
  >
  > So, before doing eigen decomposition, we have to calculate the Kernel first, then doing dimensional reduction.
  >
  > 
  >
  > Before going to the function, we also need to multiply the inverse of number of data.
  >
  > $\dfrac{1}{N}K$

  ![train_kernel_main](assets/train_kernel_main.png)

* `train.solve_by_pca` function: use Kernel PCA algorithm to find $A$ orthogonal projection matrix

  > Arguments
  > 	x: np.ndarray, Kernel matrix of data $X$
  > 	kernel_type: Literal["rbf", "linear", "poly"], used for checking some requirements
  > 	num_components: keep first k principal components
  >
  > In Part3, we only focus on Kernel PCA.
  >
  > 
  >
  > According to the normal PCA eigenvalue problem in feature space,
  > we could use eigen decomposition on centered Kernel matrix to get the first k largest eigenvectors (principal components) as $A$ matrix
  >
  > $Ka = \lambda Na$
  >
  > $\dfrac{1}{N}Ka = \lambda a$

  ![image-20230614172005560](assets/train_pca.png)

* `train.calculate_between_and_with_class_covariance` function: calculate between-class and within-class kernel covariance

  > Arguments
  > 	x: np.ndarray, Kernel matrix of data $X$
  > 	labels: np.ndarray, gt labels of data $X$
  > 	kernel_type: Literal["rbf", "linear", "poly"], used for checking some conditions
  > 	num_components: keep first k principal components
  >
  > In Part3, we only focus on Kernel LDA.
  >
  > 
  >
  > Here, it corresponds to the second and third step of Kernel LDA.
  >
  > 1. Within-class scatter $\bold{S}_W$
  >
  >    $\bold{M}_j = \dfrac{1}{N_j} \sum_{k \in C_j} k(x_k, x_i), i \in N$
  >
  >    $\bold{S}_W = \sum_{j=1}^k K_j (I - 1_{n_j}) K_j^T$
  >
  > 2. Between-class scatter $\bold{S}_B$
  >
  >    $\bold{M} = \dfrac{1}{N} \sum_{k \in N} k(x_k, x_i), i \in N$
  >
  >    $\bold{S}_B = \sum_{j=1}^k N_j (\bold{M}_j - \bold{M}) (\bold{M}_j - \bold{M})^T$

  ![image-20230614173702729](assets/train_lda_cov.png)

* `train.solve_by_fisher` function: use Kernel LDA algorithm to find $A$ orthogonal projection matrix

  > Arguments
  > 	x: np.ndarray, Kernel matrix of data $X$
  > 	labels: np.ndarray, gt labels of data $X$
  > 	kernel_type: Literal["rbf", "linear", "poly"], used for checking some conditions
  > 	num_components: keep first k principal components
  >
  > In Part3, we only focus on Kernel LDA.
  >
  > 
  >
  > After calculating the between-class and within-class kernel covariance,
  >
  > use eigen decomposition on $\bold{S}^{-1}_W\bold{S}_B$ to get the first k largest eigenvectors (principal components) as $A$ matrix.

  ![image-20230614173242875](assets/train_lda.png)

* `test.py` main: program entry point

  ![image-20230615224602171](assets/test_main.png)

* `test.evaluate_kernel` function: evaluate performance by K-NN algorithm on Kernel PCA/LDA features

  > Arguments
  > 	train_data: np.ndarray, shape (N, D), train data samples
  > 	train_labels: np.ndarray, shape (N,), train data labels
  > 	test_data: np.ndarray, shape (N, D), test data samples
  > 	test_labels: np.ndarray, shape (N,), test data labels
  > 	pca_weights: Kernel PCA weights, shape (D, E)
  > 	fisher_weights: Kernel LDA weights, shape (D, E)
  > 	k_neighbors: int, k-NN
  >
  > 
  >
  > Before classify the test data, we have to calculate kernel matrix.

  ![image-20230615224658184](assets/test_evaluate_kernel.png)

* `test.classify` function: use K-NN algorithm to classify images

  > Arguments
  > 	train_data: np.ndarray, shape (N, N), kernel matrix of train data
  > 	train_labels: np.ndarray, shape (N,), train data labels
  > 	test_data: np.ndarray, shape (K, N), kernel matrix of test data
  > 	weights: Kernel PCA or LDA weights, shape (N, E)
  > 	k_neighbors: int, k-NN

  ![image-20230614175944578](assets/test_classify.png)

### t-SNE

#### Part1

* Algorithms
  * Symmetric SNE
    1. In practice, we calculate 
  * t-SNE

#### Part2

#### Part3

#### Part4

## Experiments settings and results & Discussion

### Kernel Eigenfaces

#### Part1

* Eigenfaces

  ![eigenfaces](assets/eigenfaces.jpg)

* Fisherfaces

  ![fisherfaces](assets/fisherfaces.jpg)

* Reconstructed faces from eigenfaces

  ![original_reconstructed_eigenfaces](assets/original_reconstructed_eigenfaces.jpg)

  ![reconstructed_eigenfaces](assets/reconstructed_eigenfaces.jpg)

* Reconstructed faces from fisherfaces

  ![original_reconstructed_fisherfaces](assets/original_reconstructed_fisherfaces.jpg)

  ![reconstructed_fisherfaces](assets/reconstructed_fisherfaces.jpg)

#### Part2

K-NN algorithm: PCA, LDA (Fisher)

![image-20230615230808921](assets/knn_pca_lda.png)

#### Part3

##### Observations

* Without kernel, it is better than others with kernel.

* LDA algorithm performs best.

* It is possible that the kernel function need more finetuning, e.g. grid search, to find the best parameters.

* As you can see the results with linear and polynomial kernel, they are very similar.

  It makes sense, because the linear kernel is a part of polynomial kerel.

* The results with RBF kernel is better than other two kernels'.

  It means that the data in feature space is very likely a gaussian distribution.

##### Results

* RBF Kernel

  ![image-20230615231106059](assets/knn_rbf.png)

* Linear Kernel

  ![image-20230615231210925](assets/knn_linear.png)

* Polynomial Kernel

  ![image-20230615231255868](assets/knn_poly.png)

### t-SNE

#### Part1

##### Observations

From the results of s-SNE and t-SNE, we can know that the s-SNE has crowded problem (data points with different labels are highly overlapped).

So, in the low dimensional space, it proves that t-SNE uses t-distribution to formulate the data points better than s-SNE's.

##### Results

* Symmetric SNE

  ![s-sne_final](assets/output_20/s-sne_final.png)

* t-SNE

  ![t-sne_final](assets/output_20/t-sne_final.png)

#### Part2

* s-SNE ([imgur](https://i.imgur.com/z9Nf4se.gif))

  ![s-sne_history](assets/output_20/s-sne_history.gif)

* t-SNE ([imgur](https://i.imgur.com/b3Z7cvK.gif))

  ![t-sne_history](assets/output_20/t-sne_history.gif)

#### Part3

* s-SNE (too crowded in low dimensional space!)

  ![s-sne_similarities](assets/output_20/s-sne_similarities.png)

* t-SNE

  ![t-sne_similarities](assets/output_20/t-sne_similarities.png)

#### Part4

##### Observations

* In these points figures, we could know

  * when the perplexity is low (entropy is low), the variance of each group is very small.
  * when the perplexity is high (entropy is high), the variance of each group is very wide.
  * It proves that the perplexity will affect the choose of variance.

* The t-SNE figures always have a good ability to separate groups no matter what the perplexity is.

* From these s-SNE figures, it is hard to see any difference between them.

  But if we look from the similarity figures, we could see the difference.

  For instance, the perplexity is 5 and 35.

  **Look at the low dimensional space, because of the crowded problem, we could see the data points are very close.**

  **When the perplexity is going higher, the variance is wider. It is very obvious that the crowded problem is going worse. **

  ![s-sne_similarities](assets/output_5/s-sne_similarities.png)

  ![s-sne_similarities](assets/output_35/s-sne_similarities.png)

##### Results

* Perplexity = 5 ([imgur](https://i.imgur.com/BzsPILX.gif), [imgur](https://i.imgur.com/RpnYHgC.gif))

  ![s-sne_final](assets/output_5/s-sne_final.png)

  ![t-sne_final](assets/output_5/t-sne_final.png)

* Perplexity = 10 ([imgur](https://i.imgur.com/5fNWf5c.gif), [imgur](https://i.imgur.com/NL0gxUD.gif))

  ![s-sne_final](assets/output_10/s-sne_final.png)

  ![t-sne_final](assets/output_10/t-sne_final.png)

* Perplexity = 15 ([imgur](https://i.imgur.com/aw11rhB.gif), [imgur](https://i.imgur.com/huOPB2o.gif))

  ![s-sne_final](assets/output_15/s-sne_final.png)

  ![t-sne_final](assets/output_15/t-sne_final.png)

* Perplexity = 25 ([imgur](https://i.imgur.com/6vJm9io.gif), [imgur](https://i.imgur.com/XwC3oi0.gif))

  ![s-sne_final](assets/output_25/s-sne_final.png)

  ![t-sne_final](assets/output_25/t-sne_final.png)

* Perplexity = 30 ([imgur](https://i.imgur.com/V3J5pA9.gif), [imgur](https://i.imgur.com/dMzMyAd.gif))

  ![s-sne_final](assets/output_30/s-sne_final.png)

  ![t-sne_final](assets/output_30/t-sne_final.png)

* Perplexity = 35 ([imgur](https://i.imgur.com/ddSrH6z.gif), [imgur](https://i.imgur.com/7uYOMh4.gif))

  ![s-sne_final](assets/output_35/s-sne_final.png)

  ![t-sne_final](assets/output_35/t-sne_final.png)

## Observations and discussion

* Coming soon...