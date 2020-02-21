# -----------------------------------------------------------#
# (C) 2020 Matthew Cann
# Released under MIT Public License (MIT)
# email mcann@uwaterloo.ca
# -----------------------------------------------------------


#.......................................................................IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.linalg as la
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time

#.....................................................................CONSTANTS
nstd = 1

#.....................................................................FUNCTIONS

def classify_ij(i,j, X_data, cov_list, p_list, mean_list):
    '''Classifies the data set with respect to two classes i and j'''
    #i and j are the two classes to be classified
    #X_data is the data set being classified
    
    #cov_list is the list of covariance matrices for all classes
    #p_list is the list of priors for all classes
    #mean_list is the list of means for all classes
    
    det_cov_i = la.det(cov_list[i-1])
    det_cov_j = la.det(cov_list[j-1])
    
    pi = p_list[i-1]
    pj = p_list[j-1]
    
    mean_i = mean_list[i-1]
    mean_j = mean_list[j-1]
    #....................................................
    ML_rule = np.log(det_cov_i / det_cov_j)
    MAP_rule = np.log(det_cov_i  / det_cov_j) + (2 * np.log(pj / pi))
    
    #Normalized X
    X_norm_i = X_data - mean_i
    X_norm_j = X_data - mean_j

    #Inverse Covarience Matrix
    inv_cov_i = la.inv(cov_matrix_list[i-1])
    inv_cov_j = la.inv(cov_matrix_list[j-1])
    
    #This term is the first multiplation in the g forumla
    term_i = np.matmul(X_norm_i, inv_cov_i)
    term_j = np.dot(X_norm_j, inv_cov_j)
    
    g_i = []
    g_j = []
    for index in range(X_norm_i.shape[0]):
        g_i.append(np.dot(X_norm_i[index], term_i[index]))
        g_j.append(np.dot(X_norm_j[index], term_j[index]))
        
    condition = np.array(g_j) - np.array(g_i)

    class_ml_ij = []
    class_map_ij = []
    
    for diff in condition:
        class_ml_ij.append(int(diff > ML_rule)) # if diff>ML_rule then its class i or 1. 
        class_map_ij.append(int(diff > MAP_rule)) 
        
    return (class_ml_ij,  class_map_ij)

def predict(ml_12, ml_23, ml_31, map_12, map_23, map_31):
    '''Returns the predicted y values for three classes by comparing the 
    classified data of each pair of datasets for both ML and MAP methods'''
    
    #ML predict
    y_pred_ml = []
    for i in range(len(ml_12)):
        if ml_12[i] == 1 and ml_31[i] == 0:# Condition associated with class 1
            y_pred_ml.append(1)
        if ml_12[i] == 0 and ml_23[i] == 1:# Condition associated with class 2
            y_pred_ml.append(2)
        if ml_23[i] == 0 and ml_31[i] == 1:# Condition associated with class 3
            y_pred_ml.append(3)

    #MAP predict
    y_pred_map = []
    for i in range(len(map_12)):
        if map_12[i] == 1 and map_31[i] == 0: # Condition associated with class 1
            y_pred_map.append(1)
        if map_12[i] == 0 and map_23[i] == 1: # Condition associated with class 2
            y_pred_map.append(2)
        if map_23[i] == 0 and map_31[i] == 1:# Condition associated with class 3
            y_pred_map.append(3)
            
    return (y_pred_ml,y_pred_map)

def propabilty_error(y_true, y_predict,prior_list):
    error_1 = 0
    error_2 = 0
    error_3 = 0

    for yy, y_pred in zip(y_true, y_predict):
        if yy != y_pred:
            if y_pred == 1:
                error_1+=1
            if y_pred == 2:
                error_2+=1
            if y_pred == 3:
                error_3+=1
            
    p_e1 = error_1/600#(error_1+error_2+error_3)
    p_e2 = error_2/2100#(error_1+error_2+error_3)
    p_e3 = error_3/300#(error_1+error_2+error_3)
    print(error_1, error_2, error_3)
    
    exp_p_e = p_e1+p_e2+p_e3 #prior_list[0]*p_e1 + prior_list[1]*p_e2+prior_list[2]*p_e3

    return exp_p_e

def std_ellipse(nstd, mean, cov_matrix_1, ax):
    '''Returns the standard deviation curve of nstd using the mean and covariance
    of data to determine eigenvalues and vectors'''
    
    eig_val1, eig_vec1 = la.eig(cov_matrix_1)
    #Determines row in which the maximum eigen value is located
    index_1 = eig_val1.argsort()[::-1]
    #Re-orders the eigenvectors and values such that the maximum is first
    eig_val1, eig_vec1 = eig_val1[index_1], eig_vec1[:, index_1]
    #Extracts the largest eigen vector components
    vx1,vy1 = eig_vec1[0][0], eig_vec1[0][1]
    angle_1 = np.degrees(np.arctan2(vy1,vx1)) 
    width, height = 2 * nstd * np.sqrt(eig_val1)
    ellipse = Ellipse(xy = mean, width = width, height = height,angle = angle_1,edgecolor = 'k', ls = '--',fill=False)
    return ax.add_patch(ellipse)

def sample_prop(data):
    '''Recalculates the sample properties and returns the mean and covariance
    using the number of generated points'''
    mean = data.mean(axis =0)
    X1 = data-mean
    sample_cov_1 = np.dot(X1.T, X1.conj()) / (data.shape[0]-1)
    return mean, sample_cov_1 

def z_end_script():
    print ('\nProgrammed by Matt Cann \nDate: ',\
    time.ctime(),' \nEnd of processing''\n'  )
    return

#.........................................................................MAIN


#Class 1
p1 = 0.2
mean_1 = [3,2]
cov_matrix_1 = [[1,-1],[-1,2]]


#Class 2
p2 = 0.7
mean_2 = [5,4]
cov_matrix_2 = [[1,-1],[-1,2]]


#Class 3
p3= 0.1
mean_3 = [2, 5]
cov_matrix_3 = [[0.5,0.5],[0.5,3]]

cov_matrix_list = [cov_matrix_1, cov_matrix_2, cov_matrix_3]
prior_list = [p1,p2,p3]
means = [mean_1, mean_2, mean_3]

#.............................................................................
#DATA PROCESSING
nsample = 3000
data1 = np.random.multivariate_normal(mean_1, cov_matrix_1, int(nsample*p1))
data2 = np.random.multivariate_normal(mean_2, cov_matrix_2, int(nsample*p2))
data3 = np.random.multivariate_normal(mean_3, cov_matrix_3, int(nsample*p3))

#Add classes to generated data
class1 = np.zeros((len(data1),1))+1
class2 = np.ones((len(data2),1))+1
class3 = np.full((len(data3), 1), 2)+1

class_data1 = np.hstack((data1,class1))
class_data2 = np.hstack((data2,class2))
class_data3 = np.hstack((data3,class3))

dataframe = np.concatenate((class_data1, class_data2, class_data3))

X = dataframe[:,[0,1]] 
y = dataframe[:,2] 


#%%


# create meshgrid
resolution = 500 # 500x500 background pixels
X_xmin, X_xmax = np.min(X[:,0]), np.max(X[:,0]) #min and max values used in generating mesh 
X_ymin, X_ymax = np.min(X[:,1]), np.max(X[:,1])#min and max values used in generating mesh 


xx, yy = np.meshgrid(np.linspace(X_xmin, X_xmax, resolution), np.linspace(X_ymin, X_ymax, resolution)) # mesh grid sliced into resolution size
mesh = np.array([xx.ravel(), yy.ravel()]).T #Combines the xx and yy into array that is two columns

mesh_class_ml_12,  mesh_class_map_12 = classify_ij(1,2, mesh, cov_matrix_list,prior_list, means) # Classify between classes 1 and 2
mesh_class_ml_23,  mesh_class_map_23 = classify_ij(2,3, mesh, cov_matrix_list, prior_list, means) # Classify between classes 2 and 3
mesh_class_ml_31, mesh_class_map_31 = classify_ij(3,1, mesh, cov_matrix_list, prior_list, means) # Classify between classes 3 and 1


mesh_y_pred_ml, mesh_y_pred_map = predict(mesh_class_ml_12, mesh_class_ml_23, mesh_class_ml_31,mesh_class_map_12, mesh_class_map_23, mesh_class_map_31) # Predict values of y 

#Plots for decision regions
fig, (ax1, ax2) = plt.subplots(1,2, figsize = [10,6]) #figsize = [9, 5])
axs = [ax1,ax2] #Combine axis in list for iteratiion in for loop
y_preds = [mesh_y_pred_ml,mesh_y_pred_map] # Combine predicted y values for iteration in for loop


ax1.set_xlim([X_xmin, X_xmax])
ax2.set_xlim([X_xmin, X_xmax])
ax1.set_ylim([X_ymin, X_ymax])
ax2.set_ylim([X_ymin, X_ymax])


for ax, y_pred in zip(axs,y_preds) :
    
    ax.plot(mean_1[0], mean_1[1], 'o',color='orange', markeredgecolor = 'k',label = 'Class 1')
    ax.plot(mean_2[0], mean_2[1], 'o', color = 'green', markeredgecolor = 'k',label = 'Class 2')
    ax.plot(mean_3[0], mean_3[1], 'o', color = 'red',markeredgecolor = 'k', label = 'Class 3')
    
    std_ellipse(nstd, mean_1, cov_matrix_1, ax)
    std_ellipse(nstd, mean_2,cov_matrix_2, ax)
    std_ellipse(nstd, mean_3, cov_matrix_3, ax)
    
    ax.contourf(xx, yy, np.array(y_pred).reshape(resolution,resolution), alpha = 0.2, cmap = 'hsv') #, alpha=0.3, cmap='jet')
    ax.contour(xx, yy, np.array(y_pred).reshape(resolution,resolution), colors='k') #, alpha=0.3, cmap='jet')
    ax.legend()
    ax.set_frame_on(0)
    ax.set_frame_on(1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
plt.savefig('Q2parta_NEW1.png')
plt.show()

#%%

#Properties of the classes are reevaulated since there are 3000 samples. 
sample_mean_1, sample_cov_1 = sample_prop(data1)
sample_mean_2, sample_cov_2 = sample_prop(data2)
sample_mean_3, sample_cov_3 = sample_prop(data3)


#Makes lists of covarience matrices and means
sample_cov_list = [sample_cov_1, sample_cov_2, sample_cov_3]
sample_means = [sample_mean_1, sample_mean_2, sample_mean_3]

#Classiify the generated data between each other.
class_ml_12,  class_map_12 = classify_ij(1,2, X, sample_cov_list, prior_list, sample_means)
class_ml_23,  class_map_23 = classify_ij(2,3, X, sample_cov_list, prior_list, sample_means)
class_ml_31,  class_map_31 = classify_ij(3,1, X, sample_cov_list, prior_list, sample_means)

#Predicted class labels for both methods. 
y_pred_ml,y_pred_map = predict(class_ml_12, class_ml_23, class_ml_31,class_map_12, class_map_23, class_map_31)


ml_error_prob = propabilty_error(y, y_pred_ml,prior_list)
map_error_prob = propabilty_error(y, y_pred_map,prior_list)

print(ml_error_prob,map_error_prob)

#%%
mat1 = confusion_matrix(y, y_pred_ml)
true_mat = confusion_matrix(y,y)
mat2 = confusion_matrix(y, y_pred_map)

plt.figure(3)
sns.heatmap(true_mat, square=True, annot=True, cbar=False,fmt="d")
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

plt.figure(0)
sns.heatmap(mat1, square=True, annot=True, cbar=False,fmt="d")
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

plt.figure(1)
sns.heatmap(mat2, square=True,annot=True, cbar=False,fmt="d")

plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

print('Misclaffifing error rate of ML is {:0.4} %'.format(1-accuracy_score(y, y_pred_ml)))
print('Misclaffifing error rate of MAP is {:0.4} %'.format(1-accuracy_score(y, y_pred_map)))

z_end_script()
