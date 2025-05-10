import numpy as np
from typing import Callable
from typing import List


def square(x: ndarray) -> ndarray:
 '''
    Square each element in the input ndarray.
    '''
 return np.power(x, 2)



def leaky_relu(x: ndarray) -> ndarray:
 '''
    Apply "Leaky ReLU" function to each element in ndarray.
    '''
 return np.maximum(0.2 * x, x)



def deriv(func: Callable[[ndarray], ndarray],
 input_: ndarray,
 delta: float = 0.001) -> ndarray:
 '''
    Evaluates the derivative of a function "func" at every element in the
    "input_" array.
    '''
 return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

 # A Function takes in an ndarray as an argument and produces an ndarray
 Array_Function = Callable[[ndarray], ndarray]
 # A Chain is a list of functions
 Chain = List[Array_Function]

def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
 '''
    Evaluates two functions in a row, in a "Chain".
    '''
 assert len(chain) == 2, \
 "Length of input 'chain' should be 2"
 f1 = chain[0]
 f2 = chain[1]
 return f2(f1(x))

def sigmoid(x: ndarray) -> ndarray:
 '''
    Apply the sigmoid function to each element in the input ndarray.
    '''
 return 1 / (1 + np.exp(-x))


def chain_deriv_2(chain: Chain,input_range: ndarray) -> ndarray:
 '''
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    '''
 assert len(chain) == 2, \
 "This function requires 'Chain' objects of length 2"
 assert input_range.ndim == 1, \
 "Function requires a 1 dimensional ndarray as input_range"
 f1 = chain[0]
 f2 = chain[1]
 # df1/dx
 f1_of_x = f1(input_range)
 # df1/du
 df1dx = deriv(f1, input_range)
 # df2/du(f1(x))
 df2du = deriv(f2, f1(input_range))
 # Multiplying these quantities together at each point
 return df1dx * df2du