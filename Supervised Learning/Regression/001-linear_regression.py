import numpy as np

# Load data from a CSV or text file (assuming it's saved as 'data.txt' or 'data.csv')
data = np.loadtxt('data.txt', delimiter=',')

# Separate features (X) and target (y)
X = data[:, :-1]  # All columns except the last one (features)
y = data[:, -1]   # Last column (house price)

m,n = X.shape
epoch = 1000
lr = 0.001

#print("Features (X):\n", X[:5])  # Print first 5 rows
#print("Target (y):\n", y[:5])

#print("X:", X[1] @ [1,2])


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):             The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw,dj_db


#set random weight and bias
# np.random.seed(42)

# Weight = np.random.rand(n)
# Bias = np.random.rand()

Weight = [100,100]
Bias = 100

# print(Weight)
# print(Bias)

# print(np.dot( [2.2, 3] , Weight) + Bias)

#find gradient and update
for i in range(epoch):
    dw , db = compute_gradient(X,y, Weight, Bias)

    Weight -= lr*dw
    Bias -= lr*db

print("weight:", Weight)
print("Bias:", Bias)

def predict(x):
    return np.dot(x , Weight) + Bias


print(f"The house price is {predict([2.4, 3])} thousand dollars")

# print("Any NaN in X?", np.isnan(X).sum())
# print("Any NaN in y?", np.isnan(y).sum())
# print("Any Inf in X?", np.isinf(X).sum())
# print("Any Inf in y?", np.isinf(y).sum())