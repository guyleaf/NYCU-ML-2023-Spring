# Machine Learning Homework 5

[TOC]

## Environment

* OS

  ![Screenshot from 2023-05-14 21-00-19](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 21-00-19.png)

* Language: C++

* Standard: C++20

## Gaussian Process

### Code with detailed explanations

#### Libraries

Used library => Corresponding library in homework description

* Eigen => numpy
* OptimLib, autodiff => scipy.optimize
* Imgui, Implot => visualization
* Boost (for access file system only)
* OpenMP (for parallel acceleration)

#### Visualization

* **Setup**

  > Used to setup Imgui + OpenGL
  >
  > OpenGL is a backend for showing and drawing on window.

  ![Screenshot from 2023-05-14 20-50-32](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-50-32.png)

* **Draw result**

  > Arguments
  > 	title: plot title
  > 	data: training data points
  > 	f: line for Gaussian Process Regression with variance, f[0]: x, f[1]: y from means of conditional predictive distribution
  >
  > Setup Axes
  > ```C++
  > ImPlot::SetupAxes("x", "y");
  > // set x-axis range
  > ImPlot::SetupAxisLimits(ImAxis_X1, -60, 60);
  > ```
  >
  > Plot training data points
  > ````C++
  > // plot points with red color and circle
  > ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_RED, IMPLOT_AUTO, COLOR_RED);
  > ImPlot::PlotScatter("train data", data.col(0).data(), data.col(1).data(), data.rows());
  > ````
  >
  > Plot the line for Gaussian Process Regression
  >
  > ````C++
  > // 95% confidence interval == std * 2
  > VectorXd variance = 2 * f.col(2);
  > // calculate the lower bound and upper bound of y
  > VectorXd upperBound = f.col(1) + variance;
  > VectorXd lowerBound = f.col(1) - variance;
  > 
  > // plot shade filled with blue color
  > ImPlot::SetNextFillStyle(COLOR_BLUE, 0.5f);
  > // Equivalent to fill_between of matplotlib
  > ImPlot::PlotShaded("f(x)'s 95%% confidence", f.col(0).data(), upperBound.data(), lowerBound.data(), f.rows());
  > 
  > // plot line with black color
  > ImPlot::SetNextLineStyle(COLOR_BLACK);
  > ImPlot::PlotLine("f(x)'s mean", f.col(0).data(), f.col(1).data(), f.rows());
  > ````

  ![Screenshot from 2023-05-14 20-53-44](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-53-44.png)

* **Show GUI**

  > Main loop of Imgui
  >
  > Arguments
  > 	data: training data points
  > 	f: line for Gaussian Process Regression with variance, f[0]: x, f[1]: y from means of conditional predictive distribution
  > 	optimizedF: line for Optimized Gaussian Process Regression with variance, f[0]: x, f[1]: y from means of conditional predictive distribution

  ![Screenshot from 2023-05-14 21-06-18](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 21-06-18.png)

#### Part1

* **Rational quadratic kernel**

  * Formula

    $k(x, x') = \sigma^2 \cdot (1 + \frac{||x - x'||^{2}_{2}}{2 \cdot \alpha \cdot \ell^2})^{-\alpha}$

  * Main kernel function (without $||x - x'||^{2}_{2}$)

    > Arguments
    > 	diff: data of $||x - x'||^{2}_{2}$
    > 	kernelParameters: parameters of Rational quadratic kernel, $\sigma^2$, $\alpha$, $\ell$
    >
    > Part of formula
    > $k(x, x') = \sigma^2 \cdot (1 + \frac{diff}{2 \cdot \alpha \cdot \ell^2})^{-\alpha}$

    ![Screenshot from 2023-05-14 20-43-42](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-43-42.png)

  * Vector `lhs` -> Vector `rhs` kernel function

    > Arguments
    > 	lhs: vector $x$
    > 	rhs: vector $x'$
    > 	kernelParameters: parameters of Rational quadratic kernel, $\sigma^2$, $\alpha$, $\ell$
    >
    > Part of formula
    > $diff = ||\vec{x} - \vec{x}'||^{2}_{2}$

    ![Screenshot from 2023-05-14 20-43-53](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-43-53.png)

  * Vector `lhs` -> Scalar `rhs` kernel function

    > Arguments
    > 	lhs: vector $x$
    > 	rhs: scalar $x'$
    > 	kernelParameters: parameters of Rational quadratic kernel, $\sigma^2$, $\alpha$, $\ell$
    >
    > Part of formula
    > $diff = ||\vec{x} - x'||^{2}_{2}$

    ![Screenshot from 2023-05-14 20-44-09](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-44-09.png)

  * Scalar `lhs` -> Scalar `rhs` kernel function

    > Arguments
    > 	lhs: scalar $x$
    > 	rhs: scalar $x'$
    > 	kernelParameters: parameters of Rational quadratic kernel, $\sigma^2$, $\alpha$, $\ell$
    >
    > Part of formula (for convenience, create a vector contained one scalar)
    > $diff = ||x - x'||^{2}_{2}$

    ![Screenshot from 2023-05-14 20-44-01](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-44-01.png)

* **Gaussian Process Regression**

  * Formula

  * Main function

    ![Screenshot from 2023-05-14 21-30-39](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 21-30-39.png)

  * Calculate covariance of marginal distribution

    ![Screenshot from 2023-05-14 21-31-53](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 21-31-53.png)

  * 

#### Part2

### Experiments settings and results

#### Settings

* $\beta$: 5

#### Result

* Optimized parameters

  ![Screenshot from 2023-05-14 20-24-03](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-24-03.png)

* Figure

  ![Result](/home/leafying/git/NYCU-ML-2023-Spring/hw5/assets/Screenshot from 2023-05-14 20-16-01.png)

### Observations and discussion

## SVM

### Code with detailed explanations

#### Library matrix

#### Part1

#### Part2

#### Part3

### Experiments settings and results

#### Part1

#### Part2

#### Part3

### Observations and discussion