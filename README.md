# Bayesian_ML_Project
Generative &amp; Discriminative Classification of Images

In this project I used a Quadratic discriminant Analysis (QDA) and a Bayesian Linear Regression model to classify between images of dogs and frogs (both datasets can be found at https://www.cs.toronto.edu/%7Ekriz/cifar.html and contain 6000 images - 5750 for training and 250 for testing).

# Decision Boundaries
Given the dataset I would like to fit the mean of a Gaussian to this data. Suppose we use the following log-likelihood:
![image](https://github.com/user-attachments/assets/0998a1a0-76a8-49b1-a202-b43703481353)
where we have a prior over the mean:
![image](https://github.com/user-attachments/assets/96ded2c1-2927-4bd2-891c-01671d1239f3)
Using the following posterior ![image](https://github.com/user-attachments/assets/618fa4a6-a850-4f5a-ac5c-dccdf8cfcb03) I have found the MMSE estimate and have plotted the mean decision boundry ![image](https://github.com/user-attachments/assets/83f7fe0c-428c-427d-849f-a51dcae17b05).

![image](https://github.com/user-attachments/assets/fe25d3d7-4f83-47c2-90b3-0b928d7f63b5)


