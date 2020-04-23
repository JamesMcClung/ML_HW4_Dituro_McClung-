import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas
import math as m

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features


# Split X, y into train/test folds (Xtr, ytr/Xte, yte)
p = np.random.permutation(X.shape[0])
Xp, yp = X[p], y[p]

Xtr, ytr = Xp[0:m.floor((X.shape[0])/2)], yp[0:m.floor((X.shape[0])/2)]
Xte, yte = Xp[m.floor((X.shape[0])/2):], yp[m.floor((X.shape[0])/2):]



# Linear SVM
svmL = sklearn.svm.SVC(kernel = 'linear', C = 1e15)
svmL.fit(X = Xtr, y = ytr)
yhatL = svmL.decision_function(X = Xtr) # Linear kernel

# Non-linear SVM (polynomial kernel)
svmP = sklearn.svm.SVC(C = 1e15, kernel = 'poly', degree = 3)
svmP.fit(X = Xtr, y = ytr)
yhatP = svmP.decision_function(X = Xte) # Non-linear kernel

# Compute AUC
aucL = sklearn.metrics.auc(yhatL, yte)
aucP = sklearn.metrics.auc(yhatP, yte)

print(auc1)
print(auc2)