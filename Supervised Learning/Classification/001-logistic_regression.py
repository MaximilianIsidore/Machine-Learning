import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g

def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw

# Gradient descent function
def gradient_descent(X, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b


# Load datasets
train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")

# Drop unnecessary columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# Fill missing values
# for df in [train_df, test_df]:
#     df['Age'].fillna(df['Age'].median(), inplace=True)  
#     df['Fare'].fillna(df['Fare'].median(), inplace=True)  
#     df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  

# Fill missing values (without using inplace=True)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])


# Encode categorical variables
for df in [train_df, test_df]:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Normalize numerical features
for df in [train_df, test_df]:
    df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

# Split training data into X (features) and y (target)
X_train = train_df.drop(columns=['Survived']).values
y_train = train_df['Survived'].values

X_test = test_df.values  # No 'Survived' column in test set

# Initialize weights
w = np.zeros(X_train.shape[1])
b = 0.
alpha = 0.1  # Learning rate
num_iters = 1000  # Iterations

# Train the model
w, b = gradient_descent(X_train, y_train, w, b, alpha, num_iters)

# Predict function
def predict(X, w, b):
    return (sigmoid(np.dot(X, w) + b) >= 0.5).astype(int)

# Make predictions on test data
y_pred_test = predict(X_test, w, b)

# Display predictions
print("Predicted survival for test passengers:")
print(y_pred_test)

# test = pd.read_csv("titanic/final.csv")

# # Accuracy check (if test labels were available)
# if "Survived" in test.columns:
#     y_test = test_df['Survived'].values
#     accuracy = np.mean(y_pred_test == y_test) * 100
#     print(f"Model Accuracy on Test Data: {accuracy:.2f}%")

actual_data = pd.read_csv("titanic/actual_survival.csv")  # Replace with your actual data file path

#actual_data = actual_data.drop('PassengerId')

actual_survival = actual_data["Survived"].values

accuracy = np.mean(y_pred_test == actual_survival) * 100

print(f"Model Accuracy: {accuracy:.2f}%")