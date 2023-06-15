## NonlinearClas - Nonlinear classification with support vector machine (SVM)
Planetary Remote Sensing Laboratory / The Hong Kong Polytechnic University

---
### Overview
***[Support Vector Machines (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine "SVM")*** is a popular 
supervised machine learning algorithm used for classification and regression tasks. It is particularly effective in 
cases where the data is non-linearly separable. SVM works by finding an optimal hyperplane that maximally separates 
different classes in the feature space. The hyperplane is determined by a subset of training data points called 
support vectors. SVM can handle high-dimensional data and can be extended to handle non-linear problems using kernel 
functions. It is known for its ability to generalize well to unseen data and its robustness against overfitting. For 
nonlinear classification, Support Vector Machines (SVM) can utilize several approaches to handle non-linear decision 
boundaries:

1. The ***[Polynomial Kernel (poly)](https://en.wikipedia.org/wiki/Polynomial_kernel "poly")*** is a type of kernel 
function used in SVM for non-linear classification and regression tasks. It is commonly used to transform the input 
data into a higher-dimensional feature space, allowing the SVM to separate non-linearly separable data.

    The polynomial kernel function is defined as:

    $$K(x, y) = (gamma * <x, y> + coef0)^{degree}$$

    where

    ***gamma*** is a parameter that controls the influence of each individual training sample on the decision boundary. 
A higher gamma value makes the decision boundary more focused on the individual data points.
    
    ***coef0*** is an additional parameter that can shift the decision boundary. It affects the influence of the 
independent term in the polynomial kernel equation.

    ***degree*** is the degree of the polynomial function. It determines the complexity of the decision boundary. Higher degrees allow the SVM to capture more complex patterns in the data.



    <image src="/data/output/images/SVM_poly.png" width="400" alt="Polynomial Kernel (poly)">


2. The 
***[Gaussian Kernel](https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a "rbf")***, 
also known as the ***Radial Basis Function (RBF) kernel*** measures the similarity between two data points based on 
their Euclidean distance in the input space. It assigns higher weights to nearby points and lower weights to distant 
points. The kernel function takes the form of a bell-shaped curve, resembling a Gaussian distribution.

    The Gaussian kernel is defined as:

    $$K(x, y) = exp(-gamma * ||x - y||^2)$$

    where

    ***x*** and ***y*** are data points, $||x - y||^2$ represents the squared Euclidean distance between x and y, 
and ***gamma*** is a parameter that determines the width of the Gaussian curve. A smaller gamma value leads to a wider 
curve and smoother decision boundaries, while a larger gamma value results in a narrower curve and more localized 
decision boundaries.

    
    <image src="/data/output/images/SVM_rbf.png" width="400" alt="Gaussian Kernel">

3. ***[Sigmoid Kernel](https://en.wikipedia.org/wiki/Sigmoid_function "sigmoid")***: maps the data to a 
higher-dimensional space, creating an S-shaped decision boundary. The sigmoid kernel can be used in binary 
classification tasks. However, it is less commonly used compared to the polynomial and Gaussian kernels. In some 
cases, the sigmoid kernel may be replaced by other kernels that are more effective in capturing nonlinear patterns.

    The Sigmoid Kernel is defined as:

    $$K(x, y) = tanh(gamma * dot_product(x, y) + coef0)$$

    where

    ***x*** and ***y*** are input data points, ***gamma*** is a parameter that controls the influence of the dot 
product term, and ***coef0*** is an additional parameter that determines the offset of the kernel function.

---
### Procassing
This program uses an input DBF file with longitude (**x**) and latitude (**y**) coordinates. These files should be stored 
in the "**data**" folder. For example, there are two files (**class_n.dbf** and **class_s.dbf**) with different 
features on Mars.

As a result, you will get the line between the two classes. You can export it as an SHP file and open it in the GIS program, 
or export as CSV.

---
### Author
- Sergey Krasilnikov

### How to cite:
The paper where these classifications were used now in press and will be published in 2023/begining of 2024.