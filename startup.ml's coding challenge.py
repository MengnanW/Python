"""Instructions:
1.  Create a 5,000 x 5 multi dimensional array of random numbers
2.  Fit a multivariate normal joint probability distribution to this data
3.  Print out the MvNormal parameters (Mu, Sigma for each variable and the  covariance matrix)
4.  Sample 10,000 new points from this distribution and plot first two variables
"""
#Author: Mengnan Wang

import numpy as np
import matplotlib.pyplot as plt

#1. Create a 5,000 x 5 multi dimensional array of random numbers
r = np.random.random_sample((5000,5))
print("1. Create a {}multi dimensional array of random numbers:\n".format(r.shape), r)

#2. Fit a multivariate normal joint probability distribution to this data
mu = np.mean(r, axis=0)
cov = np.cov(r, rowvar=0)
sigma1 = np.cov(r[:,[0]], rowvar=0)
sigma2 = np.cov(r[:,[1]], rowvar=0)
sigma3 = np.cov(r[:,[2]], rowvar=0)
sigma4 = np.cov(r[:,[3]], rowvar=0)
sigma5 = np.cov(r[:,[4]], rowvar=0)

#3. Print out the MvNormal parameters (Mu, Sigma for each variable and the covariance matrix)
print("\n2. Fit the distribution and print out the MvNormal parameters:")
print("mu=" , mu)
print("sigma1=", sigma1)
print("sigma2=", sigma1)
print("sigma3=", sigma1)
print("sigma4=", sigma1)
print("sigma5=", sigma1)
print("covariance matrix=" , cov)

#4. Sample 10,000 new points from this distribution and plot first two variables
x = np.random.multivariate_normal(mu, cov, 10000)
print("\n3. Sample {} new points:".format(x.shape))
new = x[:,[0,1]]
print("Print and plot first two variables\n", new)
x,y = new.T
plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
