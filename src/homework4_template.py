from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm

def append_ones(X):
    '''X: nxm  array
    Appends a column of 1s at the end'''
    return np.append(X, np.array([[1]]*len(X)), axis=1)

class SVM4342 ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):
        # TODO change these -- they should be matrices or vectors
        # let w' be the weights including bias term, X1 be data with 1s appended
        # minimize 1/2 w•w 
        #   or 1/2 w' P w'
        # subject to yi(xi•w+b) ≥ 1
        #   or y*(X w + b) ≥ 1
        #   or y*(X1 w') ≥ 1
        X1 = append_ones(X)

        n = X.shape[0]
        m = X.shape[1]

        Ptemp = np.eye(m+1)[:-1] # removes bias terms from w'
        P = Ptemp.T.dot(Ptemp)
        q = np.zeros(m+1)
        G = -np.diag(y).dot(X1) # need to multiply y componentwise with X1 w', so G is weird
        h = -np.ones(n)

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        # minimizes 1/2 (xT P x) + q•x
        #   so P removes bias terms from w', q = 0
        # subject to G x ≤ h (componentwise ≤)
        #   so G = -yT X, h = -1
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        wb = sol['x']
        self.w = np.array(wb[:-1]).T
        self.b = np.array([wb[-1]])

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, X):
        return (self.w.dot(X.T) + self.b > 0) * 2 - 1

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print('my w and b:', svm4342.w, svm4342.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print('sk w and b:', svm.coef_, svm.intercept_)
    
    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print(diff)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")

if __name__ == "__main__": 
    test1()
    for seed in range(5):
        test2(seed)