# Project Summary - Generative &amp; Discriminative Classification of Images
In this project, I used Quadratic Discriminant Analysis (QDA) and a Bayesian Linear Regression model to classify images of dogs and frogs. The datasets used can be found at CIFAR-10 and contain 6,000 images—5,750 for training and 250 for testing.

# Decision Boundaries
Given the dataset I fit the mean of a Gaussian to this data. Suppose we use the following log-likelihood:
![image](https://github.com/user-attachments/assets/0998a1a0-76a8-49b1-a202-b43703481353)
where we have a prior over the mean:
![image](https://github.com/user-attachments/assets/96ded2c1-2927-4bd2-891c-01671d1239f3).
Using the posterior ![image](https://github.com/user-attachments/assets/618fa4a6-a850-4f5a-ac5c-dccdf8cfcb03) I found the Minimum Mean Squared Error (MMSE) estimate and have plotted the mean decision boundry ![image](https://github.com/user-attachments/assets/83f7fe0c-428c-427d-849f-a51dcae17b05).

<div align="center">
  <img src="https://github.com/user-attachments/assets/fe25d3d7-4f83-47c2-90b3-0b928d7f63b5" width="500">
</div>

# Generative Classification of Images
I implemented QDA to create a generative classifier for images. To do this, I fit a multivariate Gaussian to each class of images independently and then compared the likelihoods to classify them.
The appropriate conjugate prior for the covariance of a Gaussian is called the inverse Wishart distribution. This distribution models the covariance matrix of a Gaussian distribution since it is defined over positive definite (PD) matrices. In this project I assumed that: ![image](https://github.com/user-attachments/assets/d4a39ead-5786-489c-a97f-b1c999178ca8) where ![image](https://github.com/user-attachments/assets/26820df4-1c33-4c73-ad44-d050001715c2). The MMSE estimate for the Gaussian distribution is then: ![image](https://github.com/user-attachments/assets/b8f22495-baf1-4680-959a-445092245ec3).

For each class (dogs and frogs), I fit a Gaussian distribution and used the following equation to classify both the training and test sets:

![image](https://github.com/user-attachments/assets/7f75324d-8163-478a-af76-f7126b02ae22)

Below is the plot of the accuracy of QDA as a function of v:

<div align="center">
  <img src="https://github.com/user-attachments/assets/50345fe7-77e7-46c2-b23c-9942ee6661e0" width="500">
</div>

# Discriminative Classification of Images
I used a Bayesian Linear Regression model to classify images using the "classification as regression" method. To achieve this, I labeled the regression targets as ±1:

* +1 for the first class (dogs)

* -1 for the second class (frogs)

In other words, I fit the following model:

![image](https://github.com/user-attachments/assets/9aac7ec6-20b6-4f2c-89c5-9714c31e2b28)

The basis functions I used were Gaussian basis functions, with centers defined by the first M training points:
![image](https://github.com/user-attachments/assets/40257c78-ea02-4e1f-ba8a-17120f5b6ddb).

I then trained the model with M basis functions from each class and plotted the training and test accuracy as a function of M. The test accuracy when using all the training data as basis functions (M = all) is 0.88.

<div align="center">
  <img src="https://github.com/user-attachments/assets/bfb8af1f-8916-43c7-85cd-0e5086d146bf" width="500">
</div>

Finally, I plotted the 25 dog images that the model was most and least confident in classifying.

<div align="center">
  <img src="https://github.com/user-attachments/assets/72416798-2c7b-4c40-bf5e-0ebf55304345" width="500">
</div>

A notable observation is that the 25 most confidently classified images often contain a dog with white fur, and the dog is clearly distinguished from the background.

# Installation &amp; Setup
1. Clone the Repository
2. Download the CIFAR-10 Dataset from https://www.cs.toronto.edu/%7Ekriz/cifar.html
3. Execute the code found in classifiers.py

