import numpy as np
import pandas as pd
import csv as csv

path="C:/Users/chara/Downloads/pc1.csv"         # Change path here
pc1 = np.genfromtxt(path,delimiter=',') 

ck=np.array(pc1)

x=np.array(ck)
X=x[:,0]
y=np.array(ck)
Y=y[:,1]
z=np.array(ck)
Z=z[:,2]

X_mean=np.mean(X)
Y_mean=np.mean(Y)
Z_mean=np.mean(Z)
# Subtract the means from each variable
X_diff = X - X_mean
Y_diff = Y - Y_mean
Z_diff = Y - Z_mean

# Calculate the covariance matrix
cov_matrix = np.zeros((3, 3))
cov_matrix[0, 0] = np.sum(X_diff * X_diff) / (len(X) - 1)
cov_matrix[0, 1] = np.sum(X_diff * Y_diff) / (len(X) - 1)
cov_matrix[0, 2] = np.sum(X_diff * Z_diff) / (len(X) - 1)
cov_matrix[1, 0] = cov_matrix[0, 1]
cov_matrix[1, 1] = np.sum(Y_diff * Y_diff) / (len(Y) - 1)
cov_matrix[1, 2] = np.sum(Y_diff * Z_diff) / (len(Y) - 1)
cov_matrix[2, 0] = cov_matrix[0, 2]
cov_matrix[2, 1] = cov_matrix[1, 2]
cov_matrix[2, 2] = np.sum(Z_diff * Z_diff) / (len(Z) - 1)

print("The covarianvce matrix is:",cov_matrix)

# *****************************************************************************************************************************
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
j=np.argmin(eigenvalues)
normal=eigenvectors[:,j]
magnitude=np.sqrt(normal.dot(normal))
print("The direction of the normal surface is", normal)
print("The magnitude of the normal is", magnitude)


# **********************************************************************************************************************************
