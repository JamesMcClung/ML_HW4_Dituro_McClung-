# with batchc = itersize = 250 (takes a long time):
# acuL: 0.8207307431079025
# acuP: 0.805964512321145

import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas
import math as m

# Define the number of batches and iterations. This DRASTICALLY affects runtime.
BATCHC = 250
ITERSIZE = 250 # It is recommended that ITERSIZE << BATCHC

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)      # Labels
X = np.array(d.iloc[:,2:])  # Features

# Randomize the order of the Labels and Features together
p = np.random.permutation(X.shape[0])
Xp, yp = X[p], y[p]

# Partition the data into Training data and Testing data
part = m.floor((X.shape[0])/2)
Xtr, ytr = Xp[:part], yp[:part]
Xte, yte = Xp[part:], yp[part:]

# Split the training data into BATCHC partitions
Xsp, ysp = np.split(Xtr,BATCHC), np.split(ytr, BATCHC)

# Initialize yhatL and yhatP to be 
yhatL = np.zeros(ytr[0].shape)
yhatP = yhatL

# Initialize the SVMs for both the linear and polynomial case
svmL = sklearn.svm.SVC(kernel = 'linear', C = 1e15)
svmP = sklearn.svm.SVC(C = 1e15, kernel = 'poly', degree = 3)

for x in range(0,ITERSIZE):

    # Linear SVM
    print(f'Iteration: {x}')
    svmL.fit(X = Xsp[x], y = ysp[x])                    # Fit to the x-th batch
    yhatL = yhatL + svmL.decision_function(X = Xte)     # Test on the x-th batch's predictions and add to yhat

    # Non-linear SVM (polynomial kernel)
    svmP.fit(X = Xsp[x], y = ysp[x])                    # Fit to the x-th batch
    yhatP = yhatP + svmP.decision_function(X = Xte)     # Test on the x-th batch's predictions and add to yhat

# Divides both yhats element-wise by ITERC, 
# thus taking the average Prediction over the 
# number of iterations.
yhatL, yhatP = np.divide(yhatL, ITERSIZE), np.divide(yhatP, ITERSIZE)

# Compute AUC
aucL = sklearn.metrics.roc_auc_score(yte, yhatL)
aucP = sklearn.metrics.roc_auc_score(yte, yhatP)

print(f'acuL: {aucL}')
print(f'acuP: {aucP}')