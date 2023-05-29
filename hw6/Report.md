# Machine Learning Homework 6

[TOC]

## Environment

* Language: C++

* Standard: C++20

## Code with detailed explanations

### Libraries

Used library => Corresponding library in homework description

* Eigen => numpy
* OpenCV => visualization
* Boost (for access file system only)
* OpenMP (for parallel acceleration)

### Visualization

* **Main part**

  > Use `fittingHistory` which records the labels for every iterations to make a video.
  >
  > The duration of one iteration is 1s.
  >
  > 
  >
  > `cv::addWeighted` function is used to combine the clustering results (segmentation mask) with original image. 
  >
  > `drawMask` function is used to convert the labels to segmentation mask.

  ![image-20230529084516124](assets/image-20230529084516124.png)

* **Draw mask**

  > Convert the labels to segmentation mask.
  >
  > 
  >
  > `labels` contains the clustering results, from 0 to k-1.
  >
  > `cv::applyColorMap` is used to map the labels to the corresponding color to avoid duplicate colors.

  ![image-20230529084635319](assets/image-20230529084635319.png)

### Part1

* **Kernel K-Means**

  * Pseudo-code

    > Ignore weight

    ![image-20230529091400638](assets/image-20230529091400638.png)

  * Main function

    > Arguments
    > 	path: the image data folder
    > 	numberOfClusters: the number of clusters
    > 	init: the selected initialization method, random or k-means++
    > 	gamma1, gamma2: the hyper-parameter of RBF kernel

    ![image-20230529091820908](assets/image-20230529091820908.png)

  * `kmeans.h` header: k-means related classes

    > Follow the scikit-learn logic design.

    ![image-20230529093852567](assets/image-20230529093852567.png)

  * `KernelKMeans` constructor

    ![image-20230529094049565](assets/image-20230529094049565.png)

    ![image-20230529094149142](assets/image-20230529094149142.png)

  * `preprocess` function: extract RGB values and coordinates

    > Arguments
    > 	image: (H, W, 3), BGR values

    ![image-20230529092002708](assets/image-20230529092002708.png)

  * `calculateKernel` function: calculate all kernel values

    > Arguments
    > 	pixels: (N, 3), RGB values
    > 	coordinates: (N, 2), coordinates
    > 	gamma1: $\gamma_c$ scalar
    > 	gamma2: $\gamma_s$ scalar
    >
    > Formula
    >
    > <img src="assets/image-20230529092810965.png" alt="image-20230529092810965" style="zoom: 33%;" />
    >
    > $S(x)$ is the spatial information (coordinate)
    >
    > $C(x)$ is the color information (RGB)

    ![image-20230529092529881](assets/image-20230529092529881.png)

  * `rbf` function: RBF kernel

    > Arguments
    > 	x1: $x$ vector
    > 	x2: $x'$ vector
    > 	gamma: $\gamma$ scalar
    >
    > Formula
    >
    > $k(x, x') = e^{-\gamma||x - x'||^2}$

    ![image-20230529092657176](assets/image-20230529092657176.png)

  * `kernelKMeans.fit` function: fit the data

    > Previously, we already calculated the kernel first.
    >
    > So, we use the precomputed kernel values directly.
    >
    > Arguments
    > 	x: (N, N), the kernel values (gram matrix, similarity matrix)
    >
    > Steps
    >
    > 1. Pick k centers
    > 2. Calculate the cost between the data and centers
    > 3. Assign the label which has the smallest distance to the data
    > 4. Keep repeating 2, 3 step until the labels are not changed
    >
    > `fittingHistory` is used to store the labels for every iterations.
    >
    > Note: At the line 183, we do $1 - x$ to get the distance matrix.

    ![image-20230529094326970](assets/image-20230529094326970.png)

  * `initializeCenters` function: pick k centers initialized by the selected method

    > Arguments
    > 	x: (N, N) precomputed distance matrix or (N, features) data
    > 	init: the selected initialization method, random or k-means++
    > 	seed: the random seed
    > 	precomputed: x is the precomputed distance matrix or not.

    ![image-20230529095022292](assets/image-20230529095022292.png)

  * `randomInitialization` function: randomly pick k centers

    > Arguments
    > 	x: (N, N) precomputed distance matrix or (N, features) data
    > 	numberOfClusters: k clusters
    > 	seed: the random seed
    >
    > Steps
    >
    > 1. generate the sequence of indexes, 0 ~ N-1
    > 2. shuffle the sequence
    > 3. pick the top k rows as the centers

    ![image-20230529095601278](assets/image-20230529095601278.png)

  * `assignLabels` function: calculate the cost and assign labels

    > Arguments
    > 	x: (N, N), the kernel values
    >
    > Steps
    >
    > 1. Calculate the cost between the data and centers
    >
    >    <img src="assets/image-20230529100229119.png" alt="image-20230529100229119" style="zoom:33%;" />
    >
    > 2. Assign the label which has the smallest distance to the data

    ![image-20230529100108018](assets/image-20230529100108018.png)

* **Spectral Clustering**

  * 


### Part2

> Use command to control the number of clusters and other parameters.

![image-20230529100524496](assets/image-20230529100524496.png)

![image-20230529100644043](assets/image-20230529100644043.png)

### Part3

* **Kernel K-Means**: K-Means++ initialization

  > Arguments
  > 	x: (N, N) precomputed distance matrix or (N, features) data
  > 	numberOfClusters: k clusters
  > 	seed: the random seed
  >
  > Steps
  >
  > 	1. Choose one center uniformly at random among the data points.
  > 	1. For each data point x not chosen yet, compute $D(x)^2$ (the squared Euclidean distance) or use the precomputed distance matrix, the distance between x and the nearest center that has already been chosen.
  > 	1. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to $D(x)^2$. (The farthest point will be chosen.)

  ![image-20230529100758527](assets/image-20230529100758527.png)

* **Spectral Clustering**

### Part4

## Experiments settings and results & Discussion

### Part1

* **Kernel K-Means**

  > Number of clusters: 2
  >
  > Initialization method: random
  >
  > Gamma1: 0.00001
  >
  > Gamma2: 0.00001

  * image1

    * [Video](https://leafying.synology.me:8001/d/s/tlJkVjJWaNRni2U9GBLm3VRMXvfcJB3q/NzfL4IBDXvPyas-UnzDD3upk9nOfJ_Dy-RLhAOcHXeQo)

    * Final

      <img src="assets/image1.png_final.png" alt="image1.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image1.png_mask.png" alt="image1.png_mask" style="zoom:150%;" />

  * image2

    * [Video](https://leafying.synology.me:8001/d/s/tlJkV5I5EqW1G59E7RqSwnnMkDmmiOza/W82E22WX3M1DqvFoV5M-D-Rj_c3v-vIy-cLgA2unXeQo)
    * Final

    <img src="assets/image2.png_final.png" alt="image2.png_final" style="zoom:150%;" />

    * Mask

    <img src="assets/image2.png_mask.png" alt="image2.png_mask" style="zoom:150%;" />

* **Spectral Clustering**

### Part2

* **Kernel K-Means**

  > Initialization method: random
  >
  > Gamma1: 0.00001
  >
  > Gamma2: 0.00001

  * K = 3，image1

    * [Video](https://leafying.synology.me:8001/d/s/tlK2IhHPTDMoZAn2Px61kBUeD6hmG2JT/ETDRZkCVd_Z3WdlKjP8OJRMGQM-uPBxW-Mbngxm_YeQo)

    * Final

      <img src="assets/image1.png_final-1685327308332-5.png" alt="image1.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image1.png_mask-1685327316622-7.png" alt="image1.png_mask" style="zoom:150%;" />

  * K = 3，image2

    * [Video](https://leafying.synology.me:8001/d/s/tlK2IHsJK6GDrK3FPvgBQQe6hbEPTSN3/EyH7Ylc8YKtmtrpxL7kzsZfh1L-NjJ2j-S7lAuXjYeQo)

    * Final

      <img src="assets/image2.png_final-1685327375814-9.png" alt="image2.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image2.png_mask-1685327383312-11.png" alt="image2.png_mask" style="zoom:150%;" />

  * K = 4，image1

    * [Video](https://leafying.synology.me:8001/d/s/tlKDRkjAYAz9zubq5Xe1R40IA80v4N1K/WPSMmA6YA1bWgxkvl0aaLd3WO2-XRnxs-CLpATxTZeQo)

    * Final

      <img src="assets/image1.png_final-1685327482395-13.png" alt="image1.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image1.png_mask-1685327499391-15.png" alt="image1.png_mask" style="zoom:150%;" />

  * K = 4，image2

    * [Video](https://leafying.synology.me:8001/d/s/tlKDRYNcOEHJFozynh1pQvKgTj9Meg1f/iswTOVNZzI4pVDsUli7_IzAYmx9rUylI-IrogKBzZeQo)

      

    * Final

      <img src="assets/image2.png_final-1685327512239-17.png" alt="image2.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image2.png_mask-1685327519479-19.png" alt="image2.png_mask" style="zoom:150%;" />

* **Spectral Clustering**

### Part3

* **Kernel K-Means**

  I think there is no significant difference between random and k-means++.

  > Initialization method: k-means++
  >
  > Gamma1: 0.00001
  >
  > Gamma2: 0.00001

  * K = 2，image1

    * [Video](https://leafying.synology.me:8001/d/s/tlKRiZFb0gx4l9kDflaP7klb7aBSWAaF/L-hBJP7wwnJB61EqCAn2QoJuZcszh4DU-V7zgnwjbeQo)

    * Final

      <img src="assets/image1.png_final-1685328574880-21.png" alt="image1.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image1.png_mask-1685328582617-23.png" alt="image1.png_mask" style="zoom:150%;" />

  * K = 2，image2

    * [Video](https://leafying.synology.me:8001/d/s/tlKRi2bj2wFEx7MHf7L7c0aY1hxN45pI/aa6FQXex-EHP1JC5vAeVsalZBklWmpcJ-eLwAhCLbeQo)

    * Final

      <img src="assets/image2.png_final-1685328649712-25.png" alt="image2.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image2.png_mask-1685328657049-27.png" alt="image2.png_mask" style="zoom:150%;" />

  * K = 3，image1

    * [Video](https://leafying.synology.me:8001/d/s/tlKZr1qvG4ov4mDU315jSPmwJvjGdcT1/OUn_fQuKd6ZV7ZXSQ79pOWPgu26L5Wew-Bb1gKRHceQo)

    * Final

      <img src="assets/image1.png_final-1685329073108-29.png" alt="image1.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image1.png_mask-1685329080046-31.png" alt="image1.png_mask" style="zoom:150%;" />

  * K = 3，image2

    * [Video](https://leafying.synology.me:8001/d/s/tlKZqYtpzEq77o7rFST7oy2twnltK81y/9VVb0XfNRIntOwWx5em7LTGGIC7CrFwD-Hr2gqxjceQo)

    * Final

      <img src="assets/image2.png_final-1685329087572-33.png" alt="image2.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image2.png_mask-1685329094328-35.png" alt="image2.png_mask" style="zoom:150%;" />

  * K = 4，image1

    * [Video](https://leafying.synology.me:8001/d/s/tlKrFasoosdXs3yXmmjXxz7aCiwdKt3x/jo9xz996PqUXZbHKei9UE9MswtxtDSMh-W70AkTHceQo)

    * Final

      <img src="assets/image1.png_final-1685329135599-37.png" alt="image1.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image1.png_mask-1685329145627-39.png" alt="image1.png_mask" style="zoom:150%;" />

  * K = 4，image2

    * [Video](https://leafying.synology.me:8001/d/s/tlKrGL9R1pJrtQt7mPhl2B5O2515Bl8p/b_aZCx7aSNm5UDUgm5jsejmxaMyX1ATz-dL2gyzfceQo)

    * Final

      <img src="assets/image2.png_final-1685329152172-41.png" alt="image2.png_final" style="zoom:150%;" />

    * Mask

      <img src="assets/image2.png_mask-1685329158859-43.png" alt="image2.png_mask" style="zoom:150%;" />

* **Spectral Clustering**

### Part4

## Observations and discussion

* Coming soon...
  
