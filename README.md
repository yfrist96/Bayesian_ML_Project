# Project Summary - Generative &amp; Discriminative Classification of Images
In this project I used a Quadratic discriminant Analysis (QDA) and a Bayesian Linear Regression model to classify between images of dogs and frogs (both datasets can be found at https://www.cs.toronto.edu/%7Ekriz/cifar.html and contain 6000 images - 5750 for training and 250 for testing).

# Decision Boundaries
Given the dataset I would like to fit the mean of a Gaussian to this data. Suppose we use the following log-likelihood:
![image](https://github.com/user-attachments/assets/0998a1a0-76a8-49b1-a202-b43703481353)
where we have a prior over the mean:
![image](https://github.com/user-attachments/assets/96ded2c1-2927-4bd2-891c-01671d1239f3).
Using the following posterior ![image](https://github.com/user-attachments/assets/618fa4a6-a850-4f5a-ac5c-dccdf8cfcb03) I have found the MMSE estimate and have plotted the mean decision boundry ![image](https://github.com/user-attachments/assets/83f7fe0c-428c-427d-849f-a51dcae17b05).

![image](https://github.com/user-attachments/assets/fe25d3d7-4f83-47c2-90b3-0b928d7f63b5)

# Generative Classification of Images
I have used a QDA to create a generitave classifier for images. To do so I have fit a multivariate Gaussian to each class of images independently and then compared the likelihoods in prder to classify between them. The appropriate
conjugate prior for the covariance of a Gaussian is called the inverse Wishart distribution. The inverse-Wishart distribution is a distribution over PD matrices, which makes it perfect for modeling the covariance of a Gaussian distribution. In this project I assumed that ![image](https://github.com/user-attachments/assets/d4a39ead-5786-489c-a97f-b1c999178ca8) where ![image](https://github.com/user-attachments/assets/26820df4-1c33-4c73-ad44-d050001715c2). The MMSE estimate for the Gaussian distribution is then: ![image](https://github.com/user-attachments/assets/b8f22495-baf1-4680-959a-445092245ec3).

For each class (dogs and frogs) I have fit a Gaussian and used the following equation to classify both the training and test sets.
![image](https://github.com/user-attachments/assets/7f75324d-8163-478a-af76-f7126b02ae22)

Bellow is the plot of the accuracy of the QDA as a function of v.

![image](https://github.com/user-attachments/assets/50345fe7-77e7-46c2-b23c-9942ee6661e0)






# Discriminative Classification of Images


