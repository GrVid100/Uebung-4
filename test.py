import numpy as np
from numpy.linalg import inv

x = np.ones(8)
x = np.reshape(x, (2, 4))
x[1, 2] *= 2
x[1, 1] *= 3
x[0, 1] = 4

x_transposed = x.T

x_mul = np.matmul(x_transposed, x)
#x_inv = inv(x_mul)

#result = np.matmul(x_inv, x_transposed)

print(x)
print("-----------")

print(x_transposed)

print("-----------")
print(x_mul)

#print(x_inv.shape)
print(x_transposed.shape)