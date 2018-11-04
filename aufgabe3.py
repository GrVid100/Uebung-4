import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

training_set = pd.read_csv("TrainingSet1D.csv", names=['x_vals', 'y_vals'])
test_set = pd.read_csv("TestSet1D.csv", names=['x_vals', 'y_vals'])

default_degree = 17
default_noise = 0.015

def build_matrix(dataset, degree=default_degree):
    matrix_size = (degree + 1) * dataset.size
    X = np.ones(matrix_size)
    X = np.reshape(X, (dataset.size, degree + 1))
    for i in range(degree + 1):
        X[:, i] *= dataset ** i
    return X


def calculate_MLE_coefficients(degree=default_degree):
    X = build_matrix(training_set['x_vals'], degree)
    X_transposed = X.T
    X_transposed_mul_X = np.matmul(X_transposed, X)
    X_transposed_mul_x_inv = inv(X_transposed_mul_X)
    first_part_of_equation = np.matmul(X_transposed_mul_x_inv, X_transposed)
    coefficients = np.matmul(first_part_of_equation, training_set['y_vals'])
    return coefficients


def calculate_RIDGE_coefficients(degree=default_degree, noise=default_noise):
    X = build_matrix(training_set['x_vals'], degree)
    X_transposed = X.T
    X_transposed_mul_X = np.matmul(X_transposed, X)
    X_transposed_mul_X_plus_regulated = noise * np.identity(X_transposed_mul_X.shape[1]) + X_transposed_mul_X
    X_transposed_mul_X_plus_regulated_inv = inv(X_transposed_mul_X_plus_regulated)
    first_part_of_equation = np.matmul(X_transposed_mul_X_plus_regulated_inv, X_transposed)
    coefficients = np.matmul(first_part_of_equation, training_set['y_vals'])
    return coefficients


def calculate_error(dataset, coefficients, degree=default_degree):
    X = build_matrix(dataset['x_vals'], degree)
    Xw = np.matmul(X, coefficients)
    Xw_minus_y = Xw - dataset['y_vals']
    Xw_minus_y_transposed = Xw_minus_y.T
    multiplied_matrix = np.matmul(Xw_minus_y_transposed, Xw_minus_y)
    MSE = multiplied_matrix/dataset['x_vals'].size
    return MSE


def determine_y_value(x, coefficients):
    y = 0
    power = 0
    for coefficient in coefficients:
        y += coefficient * x ** power
        power += 1
    return y


fig, ax = plt.subplots(3, 2)

# Make plots
x_values_for_plots = np.linspace(-5, 5, 100)

y_values_for_plot_training_MLE = [determine_y_value(x, calculate_MLE_coefficients()) for x in
                                  x_values_for_plots]
ax[0, 0].scatter(training_set['x_vals'], training_set['y_vals'], color='orange')
ax[0, 0].plot(x_values_for_plots, y_values_for_plot_training_MLE)
ax[0, 0].set_title('Training Set with MLE Regression')

y_values_for_plot_test_MLE = [determine_y_value(x, calculate_MLE_coefficients()) for x in x_values_for_plots]
ax[0, 1].scatter(test_set['x_vals'], test_set['y_vals'], color='brown')
ax[0, 1].plot(x_values_for_plots, y_values_for_plot_test_MLE)
ax[0, 1].set_title('Test Set with MLE Regression')

y_values_for_plot_training_RIDGE = [determine_y_value(x, calculate_RIDGE_coefficients()) for x in
                                    x_values_for_plots]
ax[1, 0].scatter(training_set['x_vals'], training_set['y_vals'], color='orange')
ax[1, 0].plot(x_values_for_plots, y_values_for_plot_training_RIDGE)
ax[1, 0].set_title('Training Set with RIDGE Regression')

y_values_for_plot_test_RIDGE = [determine_y_value(x, calculate_RIDGE_coefficients()) for x in
                                x_values_for_plots]
ax[1, 1].scatter(test_set['x_vals'], test_set['y_vals'], color='brown')
ax[1, 1].plot(x_values_for_plots, y_values_for_plot_test_RIDGE)
ax[1, 1].set_title('Test Set with RIDGE Regression')

errors_MLE = []
for degree in range(0, 20, 1):
    error_MLE = calculate_error(test_set, calculate_MLE_coefficients(degree), degree)
    errors_MLE.append(error_MLE)

ax[2, 0].set_title('Error in Relation to Polynomial Degree')
ax[2, 0].plot(range(0, 20), errors_MLE, marker='o')

plt.show()


errors_RIDGE = []
for degree in range(2, 20, 1):
    err = []
    for epsilon in range(0, 30):
        error_RIDGE = calculate_error(test_set, calculate_RIDGE_coefficients(degree, epsilon/20), degree)
        err.append(error_RIDGE)
    errors_RIDGE.append(err)


fig2 = plt.figure()
ax2 = plt.axes(projection='3d')

x_axis = []
y_axis = []
z_axis = []
degree = 0
for deg in errors_RIDGE:
    val = 0
    for value in deg:
        x_axis.append(degree)
        z_axis.append(value)
        y_axis.append(val/100)
        val += 1
    degree += 1

plt.xlabel('Degree')
plt.ylabel('Epsilon')

ax2.scatter3D(x_axis, y_axis, z_axis)
plt.show()



print("MSE for MLE and training Set is:")
print(calculate_error(training_set, calculate_MLE_coefficients()))

print("MSE for MLE and test Set is:")
print(calculate_error(test_set, calculate_MLE_coefficients()))

print("\nMSE for RIDGE and training Set is:")
print(calculate_error(training_set, calculate_RIDGE_coefficients()))

print("MSE for RIDGE and test Set is:")
print(calculate_error(test_set, calculate_RIDGE_coefficients()))

# Plot MSE in relation to polynomial degree

