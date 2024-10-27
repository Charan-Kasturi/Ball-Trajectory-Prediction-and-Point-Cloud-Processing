# ***************************************************************STANDARD LEAST SQUARE***************************************************************
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from mpl_toolkits.mplot3d import Axes3D
import random

path1="C:/Users/chara/Downloads/pc1.csv"         # Change path here
pc1 = np.genfromtxt(path1,delimiter=',') 
path2="C:/Users/chara/Downloads/pc2.csv"         # Change path here
pc2 = np.genfromtxt(path2,delimiter=',') 
data=np.concatenate([pc1,pc2])
print(data)

x1=data[:,0] 
x2=data[:,1]
# print(x1)
# print(x2)
x3=np.ones_like(x1)
E=np.vstack((x1,x2,x3)).T
E_T=E.T
print(E_T)
G=np.vstack(data[:,2])
print(G)
# equation=E_Tx+Fy+c-G_T=0
E_T_inv=np.linalg.inv(np.dot(E_T,E))
R=np.dot(E_T_inv,E_T)
F=np.dot(R,G)
print(F)

i=F[0]
j=F[1]
k=F[2]

a=data[:,0]
b=data[:,1]
c=data[:,2]

x = np.linspace(min(a),max(a), 10)
y = np.linspace(min(b), max(b), 10)
x, y = np.meshgrid(x, y)
z = (i*x + j*y + k)
print(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot
ax.plot_surface(x, y, z,color='r',alpha=0.3)
ax.scatter(a,b,c)


# Set the axis labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()
 
# *************************************************************************TOTAL LEAST SQUARE******************************************************************
x1_mean=np.mean(x1)
x2_mean=np.mean(x2)
x3=G
x3_mean=np.mean(x3)
# print("x3:",x3)
x1_difference=x1-x1_mean
x2_difference=x2-x2_mean
x3_difference=x3-x3_mean
# Calculate the covariance matrix
covar_matrix = np.zeros((3, 3))
covar_matrix[0, 0] = np.sum(x1_difference * x1_difference) / (len(x1) - 1)
covar_matrix[0, 1] = np.sum(x1_difference * x2_difference) / (len(x1) - 1)
covar_matrix[0, 2] = np.sum(x1_difference * x3_difference) / (len(x1) - 1)
covar_matrix[1, 0] = covar_matrix[0, 1]
covar_matrix[1, 1] = np.sum(x2_difference * x2_difference) / (len(x2) - 1)
covar_matrix[1, 2] = np.sum(x2_difference * x3_difference) / (len(x2) - 1)
covar_matrix[2, 0] = covar_matrix[0, 2]
covar_matrix[2, 1] = covar_matrix[1, 2]
covar_matrix[2, 2] = np.sum(x3_difference * x3_difference) / (len(x3) - 1)
# print("The covarianvce matrix is:",covar_matrix)
eigenvalues, eigenvectors = np.linalg.eig(covar_matrix)
e=np.argmin(eigenvalues)
vectors=eigenvectors[:,e]
# print(vectors)
q=vectors[0]
w=vectors[1]
e=vectors[2]
r=np.column_stack((q,w,e))
mean=np.array([x1_mean,x2_mean,x3_mean]).reshape((-1,1))
print(mean)
print(r)
t=np.dot(r,mean)
print(t)
# equation=Qx+Wy+Ez+t=0
Q=q/e
W=w/e
E=e
T=t/e
v= np.linspace(min(a),max(a), 10)
k = np.linspace(min(b), max(b), 10)
v,k = np.meshgrid(v,k)
E= (-Q*v - W*k - T)
print(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Create a scatter plot
ax.plot_surface(v, k, E,color='r',alpha=0.3)
ax.scatter(a,b,c)


# Set the axis labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()

# ************************************************************RANSAC************************************************************************
min_points=3
Treshhold=0.20
no_iterations=300
md=np.empty((0,3))
for i in range (no_iterations):
    random_indices = data[(np.random.choice(len(data)),3), :]
    # print(random_indices)
    Mean=np.mean(random_indices)
    aa=random_indices[:,0]
    bb=random_indices[:,1]
    cc=random_indices[:,2]
    aa_diff=aa-Mean
    bb_diff=aa-Mean
    cc_diff=cc-Mean
    covari_matrix=np.zeros((3,3))
    covari_matrix = np.zeros((3, 3))
    covari_matrix[0, 0] = np.sum(aa_diff*aa_diff) / (3 - 1)
    covari_matrix[0, 1] = np.sum(aa_diff*bb_diff) / (3 - 1)
    covari_matrix[0, 2] = np.sum(aa_diff * cc_diff) / (3 - 1)
    covari_matrix[1, 0] = covari_matrix[0, 1]
    covari_matrix[1, 1] = np.sum(bb_diff * bb_diff) / (3 - 1)
    covari_matrix[1, 2] = np.sum(bb_diff* cc_diff) / (3 - 1)
    covari_matrix[2, 0] = covari_matrix[0, 2]
    covari_matrix[2, 1] = covari_matrix[1, 2]
    covari_matrix[2, 2] = np.sum(cc_diff * cc_diff) / (3 - 1)
    eigenvalues, eigenvectors = np.linalg.eig(covari_matrix)
    ss=np.argmin(eigenvalues)
    f=eigenvectors[:,ss].reshape(-1,1)
    u=f.T.dot(mean.reshape(-1,1))
    l=np.append(f,u,0)

    distance=np.empty((0,1))
    for point in data:
        c,k,s=point
        dist=(l[0]*c+l[1]*k+l[2]*s+l[3])/np.sqrt(l[0]**2+l[1]**2+l[2]**2)
        distance=np.append(distance,dist)
    # print(distance)
    innerpoints=np.array([data[i,:] for i in range(len(data)) if distance[i] < Treshhold])
    if len(innerpoints)>len(md):
        md=innerpoints.copy()
print(md)
aa=md[:,0]
bb=md[:,1]
cc=md[:,2]
aa_diff=aa-Mean
bb_diff=aa-Mean
cc_diff=cc-Mean
covari_matrix=np.zeros((3,3))
covari_matrix = np.zeros((3, 3))
covari_matrix[0, 0] = np.sum(aa_diff*aa_diff) / (3 - 1)
covari_matrix[0, 1] = np.sum(aa_diff*bb_diff) / (3 - 1)
covari_matrix[0, 2] = np.sum(aa_diff * cc_diff) / (3 - 1)
covari_matrix[1, 0] = covari_matrix[0, 1]
covari_matrix[1, 1] = np.sum(bb_diff * bb_diff) / (3 - 1)
covari_matrix[1, 2] = np.sum(bb_diff* cc_diff) / (3 - 1)
covari_matrix[2, 0] = covari_matrix[0, 2]
covari_matrix[2, 1] = covari_matrix[1, 2]
covari_matrix[2, 2] = np.sum(cc_diff * cc_diff) / (3 - 1)
eigenvalues, eigenvectors = np.linalg.eig(covari_matrix)
ss=np.argmin(eigenvalues)
f=eigenvectors[:,ss].reshape(-1,1)
u=f.T.dot(mean.reshape(-1,1))
l=np.append(f,u,0)

print("L",l)
q=l[0]
w=l[1]
t=l[2]
e=l[3]
Q=q/e
W=w/e
# E=e
T=t/e
v= np.linspace(min(a),max(a), 10)
k = np.linspace(min(b), max(b), 10)
v,k = np.meshgrid(v,k)
E= (-Q*v - W*k - T)
print(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Create a scatter plot
ax.scatter(data[:,0],data[:,1],data[:,2])
ax.plot_surface(v, k, E,color='r',alpha=0.3)



# Set the axis labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()
