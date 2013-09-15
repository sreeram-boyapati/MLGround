import numpy as np
from scipy.optimize import fmin

def cost_function(theta, X, y):    
    m = X.shape[0]
    theta = theta.reshape(X.shape[1], 1)
    error = X.dot(theta) - y 
    J = 1/(2*m) * error.T.dot(error)  
    return J

X = np.array([[1., 1.],
              [1., 2.],
              [1., 3.],
              [1., 4.]])

y = np.array([[2],[4],[6],[8]])   
initial_theta = np.ones((X.shape[1])) * 0.01
print initial_theta.shape

# test cost_function
print cost_function(initial_theta, X, y)
# [[ 14.800675]] seems okay...

# but then error here...   
print initial_theta.shape
theta = fmin(cost_function, initial_theta, args=(X, y))

[ 0.4859444   2.8627971  -0.11007741  0.14486006  2.92973392  0.62873922
 -3.53287819 -2.88852886 -0.77104406]

 [ 0.4859444   2.8627971  -0.11007741  0.14486006  2.92973392  0.62873922
 -3.53287819 -2.88852886 -0.77104406]