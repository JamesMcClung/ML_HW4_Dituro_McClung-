import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas
import math as m

# Define the number of batches. This DRASTICALLY affects runtime.
# Projections:
#  BC   T
#  -------
#  250| 4h
#  100| 6h
# 1000| 5.5h
#   10| Indefinite

BATCHC = 400

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Randomize the order of the Labels and Features together
p = np.random.permutation(X.shape[0])
Xp, yp = X[p], y[p]

# Partition the data into Training data and Testing data
Xtr, ytr = Xp[0:m.floor((X.shape[0])/2)], yp[0:m.floor((X.shape[0])/2)]
Xte, yte = Xp[m.floor((X.shape[0])/2):], yp[m.floor((X.shape[0])/2):]

# Split the training data into BATCHC partitions
Xsp, ysp = np.split(Xtr,BATCHC), np.split(ytr, BATCHC)

# Initialize yhatL and yhatP to be 
yhatL = np.zeros(ytr[0].shape)
yhatP = yhatL

# Initialize the SVMs for both the linear and polynomial case
svmL = sklearn.svm.SVC(kernel = 'linear', C = 1e15)
svmP = sklearn.svm.SVC(C = 1e15, kernel = 'poly', degree = 3)

for x in range(0,BATCHC):

    # Linear SVM
    print(f'Got Here L: {x}')
    svmL.fit(X = Xsp[x], y = ysp[x])                    # Fit to the x-th batch
    yhatL = yhatL + svmL.decision_function(X = Xte)     # Test on the x-th batch's predictions and add to yhat

    # Non-linear SVM (polynomial kernel)
    print(f'Got Here P: {x}')
    svmP.fit(X = Xsp[x], y = ysp[x])                    # Fit to the x-th batch
    yhatP = yhatP + svmP.decision_function(X = Xte)     # Test on the x-th batch's predictions and add to yhat

# Divides both yhats element-wise by BATCHC, 
# thus taking the average Prediction over the 
# batch size.
print('Got to divide')
yhatL, yhatP = np.divide(yhatL, BATCHC), np.divide(yhatP, BATCHC)

# Compute AUC
print('Got to acu')
aucL = sklearn.metrics.roc_auc_score(yte, yhatL)
aucP = sklearn.metrics.roc_auc_score(yte, yhatP)

print(f'acu1: {auc1}')
print(f'acu2: {auc2}')