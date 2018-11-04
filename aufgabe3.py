import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

training_set = pd.read_csv("TrainingSet1D.csv", names=['x_vals', 'y_vals'])
test_set = pd.read_csv("TestSet1D.csv", names=['x_vals', 'y_vals'])

default_degree = 18
default_noise = 0.001


# Aufgabenteil a)
def build_matrix(dataset, degree=default_degree):
    matrix_size = (degree + 1) * dataset.size
    X = np.ones(matrix_size)
    X = np.reshape(X, (dataset.size, degree + 1))
    for i in range(degree + 1):
        X[:, i] *= dataset ** i
    return X


# Aufgabenteil b)
def calculate_MLE_coefficients(degree=default_degree):
    X = build_matrix(training_set['x_vals'], degree)
    X_transposed = X.T
    X_transposed_mul_X = np.matmul(X_transposed, X)
    X_transposed_mul_x_inv = inv(X_transposed_mul_X)
    first_part_of_equation = np.matmul(X_transposed_mul_x_inv, X_transposed)
    coefficients = np.matmul(first_part_of_equation, training_set['y_vals'])
    return coefficients


# Aufgabenteil c)
def calculate_error(dataset, coefficients, degree=default_degree):
    X = build_matrix(dataset['x_vals'], degree)
    Xw = np.matmul(X, coefficients)
    Xw_minus_y = Xw - dataset['y_vals']
    Xw_minus_y_transposed = Xw_minus_y.T
    multiplied_matrix = np.matmul(Xw_minus_y_transposed, Xw_minus_y)
    MSE = multiplied_matrix / dataset['x_vals'].size
    return MSE


# Aufgabenteil f)
def calculate_RIDGE_coefficients(degree=default_degree, noise=default_noise):
    X = build_matrix(training_set['x_vals'], degree)
    X_transposed = X.T
    X_transposed_mul_X = np.matmul(X_transposed, X)
    X_transposed_mul_X_plus_regulated = noise * np.identity(X_transposed_mul_X.shape[1]) + X_transposed_mul_X
    X_transposed_mul_X_plus_regulated_inv = inv(X_transposed_mul_X_plus_regulated)
    first_part_of_equation = np.matmul(X_transposed_mul_X_plus_regulated_inv, X_transposed)
    coefficients = np.matmul(first_part_of_equation, training_set['y_vals'])
    return coefficients


# Make plots
def determine_y_value(x, coefficients):
    y = 0
    power = 0
    for coefficient in coefficients:
        y += coefficient * x ** power
        power += 1
    return y


current_error_MLE_training = calculate_error(training_set, calculate_MLE_coefficients())
current_error_MLE_test = calculate_error(test_set, calculate_MLE_coefficients())
current_error_RIDGE_training = calculate_error(training_set, calculate_RIDGE_coefficients())
current_error_RIDGE_test = calculate_error(test_set, calculate_RIDGE_coefficients())


fig, ax = plt.subplots(2, 2)
fig.tight_layout()
x_values_for_plots = np.linspace(-5, 5, 100)
y_values_for_plot_training_MLE = [determine_y_value(x, calculate_MLE_coefficients()) for x in
                                  x_values_for_plots]
ax[0, 0].scatter(training_set['x_vals'], training_set['y_vals'], color='orange')
ax[0, 0].plot(x_values_for_plots, y_values_for_plot_training_MLE)
ax[0, 0].set_title('Training Set with MLE Regression')
ax[0, 0].text(0, 10, 'MSE: ' + str('%.4f' % current_error_MLE_training), style='italic',
              bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

y_values_for_plot_test_MLE = [determine_y_value(x, calculate_MLE_coefficients()) for x in x_values_for_plots]
ax[0, 1].scatter(test_set['x_vals'], test_set['y_vals'], color='brown')
ax[0, 1].plot(x_values_for_plots, y_values_for_plot_test_MLE)
ax[0, 1].set_title('Test Set with MLE Regression')
ax[0, 1].text(0, 10, 'MSE: ' + str('%.4f' % current_error_MLE_test), style='italic',
              bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

y_values_for_plot_training_RIDGE = [determine_y_value(x, calculate_RIDGE_coefficients()) for x in
                                    x_values_for_plots]
ax[1, 0].scatter(training_set['x_vals'], training_set['y_vals'], color='orange')
ax[1, 0].plot(x_values_for_plots, y_values_for_plot_training_RIDGE)
ax[1, 0].set_title('Training Set with RIDGE Regression')
ax[1, 0].text(0, 10, 'MSE: ' + str('%.4f' % current_error_RIDGE_training), style='italic',
              bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

y_values_for_plot_test_RIDGE = [determine_y_value(x, calculate_RIDGE_coefficients()) for x in
                                x_values_for_plots]
ax[1, 1].scatter(test_set['x_vals'], test_set['y_vals'], color='brown')
ax[1, 1].plot(x_values_for_plots, y_values_for_plot_test_RIDGE)
ax[1, 1].set_title('Test Set with RIDGE Regression')
ax[1, 1].text(0, 10, 'MSE: ' + str('%.4f' % current_error_RIDGE_test), style='italic',
              bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

plt.savefig('plots_with_MSE.png')
plt.show()

errors_MLE_training = []
for degree in range(0, 20, 1):
    current_error_MLE = calculate_error(training_set, calculate_MLE_coefficients(degree), degree)
    errors_MLE_training.append(current_error_MLE)


fig1, ax1 = plt.subplots(1, 2)
ax1[0].set_title('MSE and Degree - Training Set')
ax1[0].plot(range(0, 20), errors_MLE_training, marker='o')
ax1[0].set_xlabel('Degree')
ax1[0].set_ylabel('MSE')

errors_MLE_test = []
for degree in range(0, 20, 1):
    current_error_MLE = calculate_error(test_set, calculate_MLE_coefficients(degree), degree)
    errors_MLE_test.append(current_error_MLE)

ax1[1].set_title('MSE and Degree - Test Set')
ax1[1].plot(range(0, 20), errors_MLE_test, marker='o')
ax1[1].set_xlabel('Degree')
ax1[1].set_ylabel('MSE')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=1.5, wspace=None, hspace=None)
plt.savefig('MSE_plots.png')
plt.show()









